from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    print(f"Number of templates: {len(template)}")
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            #Prompt 
            prompt = prompt_eng(classname)
            #texts = [t.format(classname) for t in template]
            texts = [t.format(prompt) for t in template]
            print(texts)
            print(f"Number of prompts for class '{classname}': {len(texts)}")
            #Tokenization
            texts = clip.tokenize(texts).cuda()
            print(f"Tokenized texts shape: {texts.shape}") 
            # prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            print(f"Initial embeddings shape: {class_embeddings.shape}")
            #L2 Normalization
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            print(f"Normalized embeddings shape: {class_embeddings.shape}")
            # Average across prompts
            class_embedding = class_embeddings.mean(dim=0)
            print(f"Averaged embedding shape: {class_embedding.shape}")
            #Second L2 Normalization
            class_embedding /= class_embedding.norm()
            print(f"Final normalized embedding shape: {class_embedding.shape}")
            clip_weights.append(class_embedding)
            print(f"Length of clip_weights list: {len(clip_weights)}")

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        print(f"Final clip_weights tensor shape: {clip_weights.shape}")
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx + 1, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):

                    print(f"Batch {i + 1}:")    # Print current batch
                    print(f" - images shape: {images.shape}")  # Print shape of images
                    print(f" - target shape: {target.shape}")  # Print shape of target
                    
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    
                    # This for debugging
                    # Print first batch's first feature vector
                    if i == 0:
                        print(f"\nEpoch {augment_idx}, First batch first feature vector:")
                        print(image_features[0][:10])  # Print first 10 values
                        print(f"Feature sum: {image_features.sum()}")
                    
                    print(f" - image_features shape: {image_features.shape}")  # Print shape of features
                    
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                print(' - train_features shape: {:}',[x.shape for x in train_features])

                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                print(' - cache_keys shape after epoch {:}: {:}'.format(augment_idx + 1, [x.shape for x in cache_keys]))
           
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        print(f" - cache_keys shape (after mean): {cache_keys.shape}")
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        print(f" - cache_keys shape (after normalization): {cache_keys.shape}")
        cache_keys = cache_keys.permute(1, 0)
        print(f" - cache_keys shape (after transpose): {cache_keys.shape}")
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        print(f" - cache_values shape (after one-hot encoding): {cache_values.shape}")
        #print(cache_values)


        # Print final averaged cache keys
        print("\nFinal averaged cache_keys:")
        print("First feature vector (first 10 values):")
        print(cache_keys[0][:10])
        print(f"Cache keys shape: {cache_keys.shape}")
        print(f"Cache keys sum: {cache_keys.sum()}")
        print(f"Cache keys mean: {cache_keys.mean()}")
        print(f"Cache keys std: {cache_keys.std()}")


        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    print(f" - Loaded cache_keys shape: {cache_keys.shape}")
    print(f" - Loaded cache_values shape: {cache_values.shape}")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
                
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    else:
        best_beta, best_alpha = cfg['init_beta'], cfg['init_alpha']

    return best_beta, best_alpha

#Prompt engineering
def prompt_eng(classname):
    if classname == "normal brain activity":
        prompt = "normal brain activity"
        #prompt = "typically developed children who have no history of drug misuse, brain injuries, mental problems, epilepsy, or high-risk behaviors."
    elif classname == "abnormal brain activity": 
        prompt = "abnormal brain activity"
        #prompt = "children with Attention Deficit Hyperactivity Disorder, one of the most prevalent neuro-developmental diseases, has a variety of etiologies and manifests in childhood as hyperactivity, impulsivity, and/or inattention."

        #prompt = "attention deficit hyperactivity disorder patient's brain activity"
        #prompt = "abnormal brain activity which belong to children with Attention Deficit Hyperactivity Disorder"
    return prompt


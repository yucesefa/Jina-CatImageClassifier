from jina import DocumentArray, Executor, requests
import torch
import torchvision
import os
import urllib.request
from pathlib import Path
import torch.nn.functional as F

class EvcilHayvanSiniflandirma(Executor):
    def __init__(
            self,
            num_breeds: int = 37,
            pretrained_weights: str = 'https://raw.githubusercontent.com/Bashirkazimi/pet-breed-classification/master/files/best_model.pth',
            traversal_paths: str = '@r',
            *args,
            **kwargs            
        ):

        super().__init__(*args, **kwargs)
        self.num_breeds = num_breeds
        self.pretrained_weights = pretrained_weights
        self.model = torchvision.models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, self.num_breeds) 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.download_and_load_model_weights()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage('RGB'),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.traversal_paths = traversal_paths
        self.batch_size = 1
        self.classToBreed = {0: 'Egyptian_Mau',
                             1: 'Persian',
                             2: 'Ragdoll',
                             3: 'Bombay',
                             4: 'Maine_Coon',
                             5: 'Siamese',
                             6: 'Abyssinian',
                             7: 'Sphynx',
                             8: 'British_Shorthair',
                             9: 'Bengal',
                             10: 'Birman',
                             11: 'Russian_Blue',
                             12: 'great_pyrenees',
                             13: 'havanese',
                             14: 'wheaten_terrier',
                             15: 'german_shorthaired',
                             16: 'samoyed',
                             17: 'boxer',
                             18: 'leonberger',
                             19: 'miniature_pinscher',
                             20: 'shiba_inu',
                             21: 'english_setter',
                             22: 'japanese_chin',
                             23: 'chihuahua',
                             24: 'scottish_terrier',
                             25: 'yorkshire_terrier',
                             26: 'american_pit_bull_terrier',
                             27: 'pug',
                             28: 'keeshond',
                             29: 'english_cocker_spaniel',
                             30: 'staffordshire_bull_terrier',
                             31: 'pomeranian',
                             32: 'saint_bernard',
                             33: 'basset_hound',
                             34: 'newfoundland',
                             35: 'beagle',
                             36: 'american_bulldog'
                            }

    def download_and_load_model_weights(self):
        cache_dir = Path.home() / '.cache' / 'jina-models'
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_name = os.path.basename(self.pretrained_weights)
        model_path = cache_dir / file_name
        if not model_path.exists():
            print(f'=> download {self.pretrained_weights} to {model_path}')
            urllib.request.urlretrieve(self.pretrained_weights, model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)   

    @requests
    def Classify(self, docs: DocumentArray, **kwargs):
        with torch.inference_mode():
            for doc in docs:
                image = self.transform(doc.tensor)
                print(image.shape)
                # [3, 224, 224] -> [1, 3, 224, 224 ]
                image = torch.unsqueeze(image, 0)
                output =F.softmax(self.model(image),dim=1)
                probability, category = torch.max(output, 1)
                probability = probability.numpy().tolist()[0]
                category = category.numpy().tolist()[0]
                label = self.classToBreed[category]
                doc.tags['prob'] = probability
                doc.tags['label'] = label
 
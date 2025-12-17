import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class AIDetectorModel(nn.Module):
    def __init__(self):
        super(AIDetectorModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.backbone(x)

def predict_image(image_path, model_path='best_model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AIDetectorModel().to(device)
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        human_prob = probs[0][0].item()*100
        ai_prob = probs[0][1].item()*100
        pred_class = torch.argmax(probs, dim=1).item()
    
    result = {
        'tahmin': 'İNSAN ÜRETİMİ' if pred_class==0 else 'YAPAY ZEKA',
        'insan_olasiligi': human_prob,
        'ai_olasiligi': ai_prob,
        'guven_skoru': max(human_prob, ai_prob),
        'emoji': '📸' if pred_class==0 else '🤖'
    }
    return result

# Grafik Fonksiyonları
def save_training_plots(train_loss, val_loss, train_acc, val_acc, folder='uploads'):
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    plt.plot(train_loss, label='Eğitim Kaybı')
    plt.plot(val_loss, label='Doğrulama Kaybı')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    loss_path = os.path.join(folder, 'loss_curve.png'); plt.savefig(loss_path); plt.close()
    
    plt.figure()
    plt.plot(train_acc, label='Eğitim Doğruluğu')
    plt.plot(val_acc, label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch'); plt.ylabel('Doğruluk (%)'); plt.legend()
    acc_path = os.path.join(folder, 'acc_curve.png'); plt.savefig(acc_path); plt.close()
    
    return loss_path, acc_path

def save_confusion_matrix(y_true, y_pred, folder='uploads'):
    os.makedirs(folder, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['İnsan','AI'], yticklabels=['İnsan','AI'])
    plt.xlabel('Tahmin'); plt.ylabel('Gerçek')
    path = os.path.join(folder, 'confusion_matrix.png'); plt.savefig(path); plt.close()
    return path

def save_roc_curve(y_true, y_prob, folder='uploads'):
    os.makedirs(folder, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}'); plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(loc='lower right')
    path = os.path.join(folder, 'roc_curve.png'); plt.savefig(path); plt.close()
    return path

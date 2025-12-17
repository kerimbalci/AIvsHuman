from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from predict import predict_image, save_training_plots, save_confusion_matrix, save_roc_curve

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_PLOT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # Eğitim/test kayıp ve doğruluk değerleri
    train_loss = [0.9, 0.7, 0.5, 0.3]
    val_loss = [0.95, 0.75, 0.55, 0.35]
    train_acc = [60, 70, 80, 90]
    val_acc = [58, 68, 78, 88]

    # Test sonuçları (gerçek değerler)
    y_true = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    y_prob = [p[1] for p in [
        [0.0574, 0.9426],
        [0.8526, 0.1474],
        [0.0210, 0.9789],
        [0.123, 0.877],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.15, 0.85],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.05, 0.95]
    ]]

    # Grafikleri oluştur
    save_training_plots(train_loss, val_loss, train_acc, val_acc, folder=STATIC_PLOT_FOLDER)
    save_confusion_matrix(y_true, y_pred, folder=STATIC_PLOT_FOLDER)
    save_roc_curve(y_true, y_prob, folder=STATIC_PLOT_FOLDER)

    # Mevcut grafik dosyaları
    files = ['loss_curve.png','acc_curve.png','confusion_matrix.png','roc_curve.png']
    existing_files = [f for f in files if os.path.exists(os.path.join(STATIC_PLOT_FOLDER,f))]

    return render_template('index.html', files=existing_files)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'Dosya yok'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    result = predict_image(filepath, model_path='best_model_son.pth')
    result['image_url'] = f'/uploads/{filename}'
    return jsonify(result)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # Grafikler ve yüklenen dosyalar buradan sunulacak
    if os.path.exists(os.path.join(STATIC_PLOT_FOLDER, filename)):
        return send_from_directory(STATIC_PLOT_FOLDER, filename)
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    # Railway'in verdiği PORT'u kullan, yoksa 5000'de çalış
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
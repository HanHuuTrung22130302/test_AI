const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const analyzeButton = document.getElementById('analyzeButton');
const resultText = document.getElementById('resultText');
const categorizedImages = document.getElementById('categorizedImages');

// Hiển thị hình ảnh khi upload
imageUpload.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.innerHTML = '<img src="' + e.target.result + '" alt="Hình ảnh tải lên">';
        }
        reader.readAsDataURL(file);
    }
});

// Phân tích hình ảnh bằng API
analyzeButton.addEventListener('click', async function () {
    resultText.innerHTML = 'Đang phân tích...';

    const file = imageUpload.files[0];
    if (!file) {
        resultText.innerHTML = 'Vui lòng chọn hình ảnh trước khi phân tích!';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Có lỗi xảy ra trong quá trình phân tích hình ảnh.');
        }

        const data = await response.json();
        const result = data.prediction;
        resultText.innerHTML = 'Kết quả: ' + result;

        // Thêm hình ảnh vào mục phân loại
        const newImageBox = document.createElement('div');
        newImageBox.classList.add('image-box');
        newImageBox.innerHTML = '<img src="' + imagePreview.querySelector('img').src + '" alt="Hình ảnh đã phân loại"><p>Loại: ' + result + '</p>';
        categorizedImages.appendChild(newImageBox);
    } catch (error) {
        resultText.innerHTML = 'Có lỗi: ' + error.message;
    }
});
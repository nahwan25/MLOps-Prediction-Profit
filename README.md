# MLOps Prediction Profit

Proyek ini adalah implementasi **MLOps pipeline** untuk memprediksi profit menggunakan Python, FastAPI, dan Docker. Aplikasi ini menyediakan API yang bisa di-deploy ke cloud.

---

## ⚡ Instalasi dan Setup Lokal


Clone repository:

```bash
git clone https://github.com/nahwan25/MLOps-Prediction-Profit.git
cd MLOps-Prediction-Profit

pip install -r requirements.txt

uvicorn app.app:app --reload

docker build -t ml-api .

docker run -p 8000:8000 ml-api


Endpoint utama: /predict (POST)

Kamu bisa mengirim Request JSON berisi data fitur, lalu mendapatkan prediksi profit sebagai response.

Contoh ilustrasi request dan response:

![Contoh Request dan Response](assets/prediction.jpg)




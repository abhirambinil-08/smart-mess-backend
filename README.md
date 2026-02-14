# Smart Mess Feedback System – Backend

This is the FastAPI backend for the Smart Mess Feedback System.  
It manages feedback collection, authentication, analytics, QR generation, and AI-based rule insights.

---

## Tech Stack

- FastAPI
- MongoDB (Motor - Async Driver)
- JWT Authentication (PyJWT)
- bcrypt (Password Hashing)
- Pydantic
- qrcode
- CORS Middleware

---

## Features

- JWT-based admin authentication
- Secure password hashing
- Duplicate feedback prevention
- Confidence score validation
- Dashboard analytics (averages & totals)
- AI rule-based hygiene insights
- QR code generation for each mess
- Protected configuration APIs

---

## Database Collections

### feedback
Stores validated feedback data.

Fields:
- institution
- mess
- meal_type
- food_quality
- taste
- hygiene
- portion_size
- comment
- device_fingerprint
- confidence_score
- validated
- timestamp

### admin_users
Stores administrator accounts.

Fields:
- email
- password_hash
- role
- is_active
- created_at

### mess
Stores configured mess data.

Fields:
- name
- institution
- created_at

---

## Environment Variables

Create a `.env` file in the backend folder:

MONGO_URL=your_mongodb_connection

DB_NAME=smart_mess_db

JWT_SECRET=your_secret_key

BASE_URL=http://localhost:5173


ENV=development


---

## Run Locally

cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn server:app --reload


Backend runs on:
http://localhost:8000

Smart Mess Feedback System – Backend

This is the FastAPI backend for the Smart Mess Feedback System.

It handles feedback collection, authentication, dashboard analytics, QR generation, and AI-based rule analysis.

Tech Stack

•	FastAPI

•	MongoDB (Motor)

•	JWT Authentication

•	bcrypt (Password Hashing)

•	Pydantic

•	qrcode


Features

•	JWT-based admin authentication

•	Duplicate feedback prevention

•	Confidence score validation

•	Dashboard analytics (averages & totals)

•	AI rule-based hygiene insights

•	QR code generation for each mess

•	Protected configuration APIs


Database Collections

•	feedback

•	admin_users

•	mess


Environment Variables (.env)

MONGO_URL=your_mongodb_connection

DB_NAME=smart_mess_db

JWT_SECRET=your_secret_key

BASE_URL=http://localhost:5173

ENV=development


Run Locally

cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn server:app --reload

Backend runs on:

http://localhost:8000

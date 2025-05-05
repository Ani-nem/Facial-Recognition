# Facial Recognition System

A full-stack web application for detecting and recognizing faces in images and video streams. The system integrates computer vision models, a PostgreSQL vector database, and a modern web dashboard to manage, label, and merge detected identities.

## Features

* **Person Detection:** Uses Ultralytics YOLOv8 for real-time person detection in image directories or video streams.
* **Face Recognition:** Leverages DeepFace to compute facial embeddings and recognize known individuals.
* **Embeddings Storage:** Stores face embeddings in PostgreSQL using the pgvector extension for efficient similarity search.
* **Web Dashboard:** Built with Next.js, React, and Shadcn/UI to display live or uploaded images, list detected persons, rename and merge identities.
* **Cloud Integration:** Stores original and cropped images in AWS S3; backend hosted on AWS and secured via Auth0 or JWT authentication.

## Tech Stack

* **Backend:** Python, FastAPI, SQLAlchemy, PostgreSQL (+ pgvector), boto3 for S3
* **Frontend:** Next.js (App Router), React, TypeScript, Shadcn/UI, Tailwind CSS
* **CV Libraries:** Ultralytics YOLOv8, DeepFace, OpenCV
* **Cloud Services:** AWS S3, AWS RDS/PostgreSQL
* **Authentication:** custom JWT (moving to auth0 soon)

## Pending revisions
* Updating custom JWT to commercial Auth0
* Adding ability to rearrange falsely identified images via dragging the image into another group
* Adding ability to rename the identified person to an actual name

## Getting Started

### Prerequisites

* Node.js (v18+)
* Python (3.9+)
* PostgreSQL with pgvector extension
* AWS account with S3 bucket

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ani-nem/Facial-Recognition.git
   cd Facial-Recognition
   ```

2. **Backend Setup**

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Configure .env with DATABASE_URL and AWS credentials
   ```

3. **Database Initialization**

   ```bash
   psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"
   alembic upgrade head  # if using migrations
   ```

4. **Start Backend**

   ```bash
   uvicorn app.main:app --reload
   ```

5. **Frontend Setup**

   ```bash
   cd ../frontend
   npm install

   # Configure .env.local with NEXT_PUBLIC_API_URL, NEXT_PUBLIC_AUTH0 settings
   npm run dev
   ```

6. **Visit** `http://localhost:3000/login` to login/register, which you will then be redirected to the dashboard.

## Usage

* Upload images to start detection.
* View detected faces in the dashboard under "Unknown Faces" and assign names.
* Download cropped face images


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

# Getting Started

# Facial Recognition Application

This documentation provides a comprehensive guide to setting up and running the Facial Recognition web application.

## Overview

The Facial Recognition project is a modern web application designed for facial detection and recognition. It allows users to upload, organize, and process images, featuring user authentication, image management, and facial recognition capabilities powered by `dlib` and `Ultralytics yolov11`.

## Tech Stack

### Frontend
*   **Next.js + React**: Framework for building user interfaces.
*   **SHADcn**: UI component library.
*   **TypeScript**: Statically typed superset of JavaScript.
*   **Tailwind CSS**: Utility-first CSS framework.

### Backend
*   **Python**: Primary backend language.
*   **dlib**: Machine learning library for facial recognition.
*   **Ultralytics yolov11**: Machine learning library for object detection.
*   **AWS S3**: Cloud storage for images.

## Features

*   **User Authentication**: Secure login and protected routes.
*   **Image Upload**: Upload and process images for facial recognition.
*   **Image Organization**: Images are grouped by detected persons.
*   **Batch Operations**: Download all images for a specific person as a ZIP file.
*   **Responsive UI**: Mobile-friendly interface.

## Getting Started

### Prerequisites

Ensure you have the following installed:

*   **Node.js**: v18.17.0 or higher
*   **npm**: v9.6.7 or higher (or yarn/pnpm/bun)
*   **Python**: v3.8 or higher
*   **Conda**: Recommended for managing Python dependencies.

### Installation

Follow these steps to set up the frontend and backend services.

#### 1. Clone the Repository

```bash
git clone https://github.com/Ani-nem/Facial-Recognition.git
cd Facial-Recognition
```

#### 2. Frontend Setup

The frontend is built with Next.js and React.

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    # or yarn install
    # or pnpm install
    # or bun install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    # or yarn dev
    # or pnpm dev
    # or bun dev
    ```
    The frontend application will be accessible at `http://localhost:3000`.

#### 3. Backend Setup

The backend is built with Python.

1.  Navigate to the root directory of the project (if you are in `frontend`, go up one level):
    ```bash
    cd ..
    ```
2.  Create and activate a Conda environment using the provided script:
    ```bash
    python conda_export_pip.py > environment.yml
    conda env create -f environment.yml
    conda activate facial-recognition-env # Replace 'facial-recognition-env' with the actual environment name if different
    ```
3.  Install additional Python dependencies (if any are not covered by `environment.yml`):
    ```bash
    pip install -r requirements.txt # Assuming a requirements.txt exists or is generated
    ```
4.  Run the backend server:
    ```bash
    python backend/server.py
    ```
    The backend API will typically run on `http://localhost:5000` (or as configured in `backend/server.py`).

### Project Structure

```
.
├── .gitignore
├── README.md
├── backend/
│   ├── auth/                 # Authentication modules
│   ├── database/             # Database models and utilities
│   ├── facialrecognition.py  # Core facial recognition logic
│   ├── server.py             # Backend API entry point
│   └── ...
├── conda_export_pip.py       # Script to export conda environment
├── datasets/                 # Sample data (e.g., video.mp4)
├── environment.yml           # Conda environment definition
└── frontend/
    ├── app/                  # Next.js pages and routes
    ├── components/           # Reusable React components
    ├── hooks/                # Custom React hooks
    ├── lib/                  # Utility functions and Axios setup
    ├── public/               # Static assets
    └── ...
```
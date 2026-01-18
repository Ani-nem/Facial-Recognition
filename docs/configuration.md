# Configuration

# Ani-nem/Facial-Recognition Documentation

This documentation provides a comprehensive guide to the Facial Recognition web application, covering its features, setup, and configuration.

## Overview

The Facial Recognition project is a modern web application designed for facial detection and recognition. It enables users to upload, organize, and process images, leveraging dlib for facial recognition and Next.js for a responsive user interface.

## Features

*   **User Authentication:** Secure login and protected routes.
*   **Image Upload:** Process images for facial recognition.
*   **Image Organization:** Images grouped by detected persons.
*   **Batch Operations:** Download all images for a specific person as a ZIP file.
*   **Responsive UI:** Mobile-friendly interface.

## Tech Stack

### Frontend

*   **Next.js + React:** Server-side rendered React framework.
*   **SHADcn:** UI component library.
*   **TypeScript:** Type-safe JavaScript.
*   **Tailwind CSS:** Utility-first CSS framework.

### Backend

*   **Python:** Primary backend language.
*   **dlib:** Machine learning library for facial recognition.
*   **Ultralytics yolov11:** Machine learning library for Object Detection.
*   **AWS S3:** Cloud-based image storage.

## Getting Started

### Prerequisites

*   Node.js (and npm/yarn/pnpm/bun)
*   Python with Conda (recommended)

### Frontend Setup

1.  Navigate to the frontend directory:
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
4.  Access the application at `http://localhost:3000`.

### Backend Setup

1.  Generate and create a Conda environment:
    ```bash
    python conda_export_pip.py > environment.yml
    conda env create -f environment.yml
    ```
2.  Activate the Conda environment:
    ```bash
    conda activate facial-recognition-env # or the name you chose
    ```
3.  Run the backend server (specific command not provided in README, typically `python server.py` or similar).

## Configuration Guide

### Environment Variables

The application uses environment variables for sensitive information and configuration.

#### Frontend

| Name                 | Description                                    | Required | Default |
| :------------------- | :--------------------------------------------- | :------- | :------ |
| `NEXT_PUBLIC_API_URL`| Base URL for the backend API.                  | Yes      |         |

#### Backend

| Name                 | Description                                    | Required | Default |
| :------------------- | :--------------------------------------------- | :------- | :------ |
| `AWS_ACCESS_KEY_ID`  | AWS access key for S3.                         | Yes      |         |
| `AWS_SECRET_ACCESS_KEY`| AWS secret access key for S3.                  | Yes      |         |
| `S3_BUCKET_NAME`     | Name of the S3 bucket for image storage.       | Yes      |         |
| `DATABASE_URL`       | Connection string for the database.            | Yes      |         |
| `SECRET_KEY`         | Secret key for session management/JWT.         | Yes      |         |
| `DEBUG`              | Enables debug mode (e.g., for Flask).          | No       | `False` |

### Configuration Files

*   **`backend/auth/auth_config.py`**: Contains authentication-related settings (e.g., JWT expiry).
*   **`backend/database/db_config.py`**: Database connection parameters.
*   **`frontend/next.config.ts`**: Next.js specific configurations.
*   **`frontend/tailwind.config.ts`**: Tailwind CSS configuration.

### Build Options

#### Frontend

To build the frontend for production:

```bash
cd frontend
npm run build
# or yarn build
# or pnpm build
# or bun build
```

This will create an optimized production build in the `.next` directory.

#### Backend

The backend typically runs directly using Python. For deployment, consider using a WSGI server like Gunicorn or uWSGI.

Example with Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend.server:app
```
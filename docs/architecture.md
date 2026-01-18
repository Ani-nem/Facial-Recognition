# Architecture

# Ani-nem/Facial-Recognition Documentation

## Overview

This repository hosts a modern web application for facial recognition, integrating a Next.js frontend with a Python backend. It enables users to upload, organize, and process images for facial detection and recognition, leveraging dlib and Ultralytics YOLOv11.

## Tech Stack

### Frontend

| Technology     | Description                                |
| :------------- | :----------------------------------------- |
| **Next.js**    | React framework with server-side rendering |
| **React**      | UI library                                 |
| **SHADcn**     | UI component library                       |
| **TypeScript** | Type-safe JavaScript                       |
| **Tailwind CSS** | Utility-first CSS framework              |

### Backend

| Technology           | Description                                    |
| :------------------- | :--------------------------------------------- |
| **Python**           | Primary backend language                       |
| **dlib**             | Machine learning library for facial recognition |
| **Ultralytics YOLOv11** | Machine learning library for Object Detection |
| **AWS S3**           | Cloud-based image storage                      |

## Features

*   **User Authentication**: Secure login and protected routes.
*   **Image Upload**: Upload and process images for facial recognition.
*   **Image Organization**: Images are grouped by detected persons.
*   **Batch Operations**: Download all images for a specific person as a ZIP file.
*   **Responsive UI**: Mobile-friendly interface.

## Getting Started

### Prerequisites

*   Node.js (npm/yarn/pnpm/bun)
*   Python with Conda (recommended)

### Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install # or yarn install / pnpm install / bun install
    ```
3.  Run the development server:
    ```bash
    npm run dev # or yarn dev / pnpm dev / bun dev
    ```
    Access the application at `http://localhost:3000`.

### Backend Setup

1.  Create and activate Conda environment:
    ```bash
    python conda_export_pip.py > environment.yml
    conda env create -f environment.yml
    conda activate facial-recognition-env # (or whatever your environment is named)
    ```
2.  Install additional dependencies (if any, not explicitly in `environment.yml`):
    ```bash
    pip install -r requirements.txt # if a requirements.txt exists
    ```
3.  Run the backend server (specific command not provided in README, typically `python server.py` or similar).

## Architecture

### High-Level Overview

The application follows a client-server architecture, with a Next.js frontend communicating with a Python backend. The frontend handles user interaction and display, while the backend manages user authentication, image processing (facial recognition, detection), and data storage.

### Folder Structure

```
.
├── .gitignore
├── README.md
├── backend/
│   ├── __init__.py
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── auth_config.py
│   ├── bruh.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   ├── db_config.py
│   │   └── models.py
│   ├── facialrecognition.py
│   ├── server.py
│   ├── testing.py
│   ├── util.py
│   └── yolo11n.pt
├── conda_export_pip.py
├── datasets/
│   └── video.mp4
├── environment.yml
└── frontend/
    ├── .gitignore
    ├── README.md
    ├── app/
    │   ├── dashboard/
    │   │   └── page.tsx
    │   ├── favicon.ico
    │   ├── globals.css
    │   ├── layout.tsx
    │   ├── login/
    │   │   └── page.tsx
    │   └── page.tsx
    ├── auth/
    │   └── AuthProvider.tsx
    ├── components.json
    ├── components/
    │   ├── fileinputform.tsx
    │   ├── groupedImages.tsx
    │   ├── loginform.tsx
    │   ├── protectedRoute.tsx
    │   └── ui/
    │       ├── button.tsx
    │       ├── card.tsx
    │       ├── input.tsx
    │       ├── label.tsx
    │       ├── scroll-area.tsx
    │       ├── toast.tsx
    │       └── toaster.tsx
    ├── eslint.config.mjs
    ├── hooks/
    │   └── use-toast.ts
    ├── lib/
    │   ├── axios.ts
    │   └── utils.ts
    ├── next.config.ts
    ├── package-lock.json
    ├── package.json
    ├── postcss.config.mjs
    ├── public/
    │   ├── file.svg
    │   ├── globe.svg
    │   ├── next.svg
    │   ├── vercel.svg
    │   └── window.svg
    ├── tailwind.config.ts
    ├── tsconfig.json
    └── utils/
        └── auth.ts
```

### Key Design Patterns

*   **Client-Server Architecture**: Clear separation of concerns between the frontend UI and backend logic.
*   **Component-Based UI (React)**: Reusable UI components for modular and maintainable frontend development.
*   **API-Driven Communication**: Frontend interacts with the backend via RESTful APIs.
*   **Modular Backend**: Backend logic is organized into modules for authentication, database interaction, and facial recognition.

### Data Flow

1.  **User Interaction**: A user interacts with the Next.js frontend (e.g., logs in, uploads an image).
2.  **Frontend Request**: The frontend sends an HTTP request (e.g., POST for image upload) to the Python backend API.
3.  **Backend Processing**:
    *   The backend receives the request.
    *   Authentication middleware verifies the user's identity.
    *   For image uploads, the image is processed using dlib and YOLOv11 for facial detection and recognition.
    *   Processed images and associated metadata are stored in AWS S3 and the database.
4.  **Backend Response**: The backend sends an HTTP response (e.g., success message, processed data) back to the frontend.
5.  **Frontend Update**: The frontend receives the response and updates the UI accordingly (e.g., displays grouped images, shows a success toast).
# API Reference

# Ani-nem/Facial-Recognition

A modern web application for facial recognition, built with Next.js, React, and Python.

## Overview

This project is a facial recognition system that allows users to upload, organize, and process images for facial detection and recognition. The application features user authentication, image management, and facial recognition capabilities using dlib.

## Tech Stack

### Frontend
- **Next.js + React**: React framework with server-side rendering
- **SHADcn**: UI component library
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework

### Backend
- **Python**: Primary backend language
- **dlib**: Machine learning library for facial recognition
- **Ultralytics yolov11**: Machine learning library for Object Detection
- **AWS S3**: Cloud based Image storage

## Features

- **User Authentication**: Secure login and protected routes
- **Image Upload**: Upload and process images for facial recognition
- **Image Organization**: Images are grouped by detected persons
- **Batch Operations**: Download all images for a specific person as a ZIP file
- **Responsive UI**: Mobile-friendly interface built with Tailwind CSS

## Getting Started

### Prerequisites
- Node.js and npm/yarn/pnpm/bun
- Python with Conda (recommended for managing dependencies)

### Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    # or
    yarn install
    # or
    pnpm install
    # or
    bun install
    ```

3.  Run the development server:
    ```bash
    npm run dev
    # or
    yarn dev
    # or
    pnpm dev
    # or
    bun dev
    ```

4.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

### Backend Setup

1.  Set up a Conda environment using the provided export script:
    ```bash
    python conda_export_pip.py > environment.yml
    conda env create -f environment.yml
    ```

2.  Activate the Conda environment:
    ```bash
    conda activate <environment_name>
    ```
    (Replace `<environment_name>` with the name specified in `environment.yml` or chosen during creation.)

3.  Run the backend server:
    ```bash
    python backend/server.py
    ```

## API Reference

### Frontend Components

#### `components/fileinputform.tsx`
A component for uploading image files.

##### Props

| Prop Name | Type     | Description             |
| :-------- | :------- | :---------------------- |
| `onUpload`| `(files: File[]) => void` | Callback function triggered after files are selected. |

##### Usage Example
```typescript jsx
import { FileInputForm } from '@/components/fileinputform';

function MyUploadPage() {
  const handleUpload = (files: File[]) => {
    console.log('Selected files:', files);
    // Logic to send files to backend
  };

  return <FileInputForm onUpload={handleUpload} />;
}
```

#### `components/loginform.tsx`
Handles user login authentication.

##### Props
No direct props. Manages internal state for username/password and interacts with authentication API.

##### Usage Example
```typescript jsx
import { LoginForm } from '@/components/loginform';

function LoginPage() {
  return (
    <div>
      <h1>Login to your account</h1>
      <LoginForm />
    </div>
  );
}
```

#### `components/protectedRoute.tsx`
A Higher-Order Component (HOC) to protect routes, ensuring only authenticated users can access child components.

##### Props

| Prop Name | Type     | Description             |
| :-------- | :------- | :---------------------- |
| `children`| `React.ReactNode` | The components to render if the user is authenticated. |

##### Usage Example
```typescript jsx
import { ProtectedRoute } from '@/components/protectedRoute';
import DashboardContent from './DashboardContent';

function DashboardPage() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  );
}
```

### Backend Endpoints

The backend (`backend/server.py`) exposes RESTful APIs for authentication, image management, and facial recognition.

#### Authentication
-   **`/api/login`**: User login.
    -   **Method**: `POST`
    -   **Request Body**: `{ "username": "...", "password": "..." }`
    -   **Response**: `{ "token": "...", "user": { ... } }` on success.

#### Image Upload & Processing
-   **`/api/upload`**: Uploads images for processing.
    -   **Method**: `POST`
    -   **Request Body**: `multipart/form-data` with image files.
    -   **Response**: Status of the upload and processing.

#### Image Retrieval
-   **`/api/images`**: Retrieves all processed images, potentially grouped by person.
    -   **Method**: `GET`
    -   **Response**: Array of image data, including associated person IDs.

#### Batch Download
-   **`/api/download/person/:personId`**: Downloads all images associated with a specific person as a ZIP file.
    -   **Method**: `GET`
    -   **Response**: `application/zip` file.

### Common Patterns

#### Authentication Flow
1.  User submits credentials via `LoginForm`.
2.  Frontend sends `POST` request to `/api/login`.
3.  Backend authenticates and returns a JWT token.
4.  Frontend stores the token (e.g., in `localStorage`) and uses `AuthProvider` to manage global authentication state.
5.  `ProtectedRoute` checks for the token before rendering protected content.

#### Image Processing Flow
1.  User selects images using `FileInputForm`.
2.  Frontend sends `POST` request with `multipart/form-data` to `/api/upload`.
3.  Backend receives images, stores them in AWS S3, processes them with dlib and YOLOv11 for facial detection/recognition.
4.  Processed image metadata (including detected persons) is stored in the database.
5.  Frontend fetches grouped images for display.
# Project Overview

# Ani-nem/Facial-Recognition

## Project Overview

This project is a modern web application designed for facial recognition, enabling users to upload, organize, and process images for facial detection and recognition. It provides a comprehensive solution for managing image data with integrated facial recognition capabilities.

### Key Features

*   **User Authentication:** Secure login and protected routes for authorized access.
*   **Image Upload & Processing:** Upload images for facial detection and recognition.
*   **Image Organization:** Automatically groups images by detected persons.
*   **Batch Operations:** Download all images associated with a specific person as a ZIP file.
*   **Responsive UI:** Mobile-friendly interface built with Tailwind CSS.
*   **Cloud Storage:** Utilizes AWS S3 for robust and scalable image storage.

### Technology Stack

| Name                 | Purpose                                     |
| :------------------- | :------------------------------------------ |
| **Next.js + React**  | Frontend framework with SSR                 |
| **SHADcn**           | UI component library                        |
| **TypeScript**       | Type-safe JavaScript for frontend           |
| **Tailwind CSS**     | Utility-first CSS framework                 |
| **Python**           | Primary backend language                    |
| **dlib**             | Machine learning library for facial recognition |
| **Ultralytics yolov11** | Machine learning library for object detection |
| **AWS S3**           | Cloud-based image storage                   |

### Project Status

The project is actively developed, with core features for facial recognition, user authentication, and image management implemented. Future enhancements will focus on improving recognition accuracy and expanding batch processing capabilities.

## Getting Started

### Prerequisites

Ensure you have the following installed:

*   **Node.js**: Includes npm (or yarn/pnpm/bun)
*   **Python**: With Conda (recommended for dependency management)

### Frontend Setup

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies using your preferred package manager:
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
4.  Open [http://localhost:3000](http://localhost:3000) in your browser.

### Backend Setup

1.  Generate and create a Conda environment:
    ```bash
    python conda_export_pip.py > environment.yml
    conda env create -f environment.yml
    ```
2.  Activate the Conda environment:
    ```bash
    conda activate facial-recognition-env # Replace with your environment name if different
    ```
3.  Install additional Python dependencies if not covered by `environment.yml`:
    ```bash
    pip install -r requirements.txt # Assuming a requirements.txt exists or create one
    ```
4.  Run the backend server (specific command not provided, typically `python server.py` or similar):
    ```bash
    # Example:
    python backend/server.py
    ```

## Project Structure

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
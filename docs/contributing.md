# Contributing

# Facial Recognition

A modern web application for facial recognition, built with Next.js, React, and Python.

## Overview

This project is a facial recognition system that allows users to upload, organize, and process images for facial detection and recognition. The application features user authentication, image management, and facial recognition capabilities using dlib.

## Tech Stack

### Frontend
- **Next.js + React** - React framework with server-side rendering
- **SHADcn** - UI component library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework

### Backend
- **Python** - Primary backend language
- **dlib** - Machine learning library for facial recognition
- **Ultralytics yolov11** - Machine learning library for Object Detection
- **AWS S3** - Cloud based Image storage

## Features

- **User Authentication** - Secure login and protected routes
- **Image Upload** - Upload and process images for facial recognition
- **Image Organization** - Images are grouped by detected persons
- **Batch Operations** - Download all images for a specific person as a ZIP file
- **Responsive UI** - Mobile-friendly interface built with Tailwind CSS

## Getting Started

### Prerequisites
- Node.js and npm/yarn/pnpm/bun
- Python with Conda (recommended for managing dependencies)

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   # or
   bun install
   ```

3. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   # or
   bun dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

### Backend Setup

1. Set up a Conda environment using the provided export script:
   ```bash
   python conda_export_pip.py > environment.yml
   conda env create -f environment.yml
   ```

2. Activate the Conda environment:
   ```bash
   conda activate <environment_name>
   ```
   (Replace `<environment_name>` with the name specified in `environment.yml` or the default name if not specified.)

3. Run the backend server:
   ```bash
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

## Contributing

We welcome contributions to the Facial Recognition project!

### How to Contribute

1.  **Fork** the repository.
2.  **Clone** your forked repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/Facial-Recognition.git
    cd Facial-Recognition
    ```
3.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    # or
    git checkout -b bugfix/issue-description
    ```
4.  **Make your changes** and test them thoroughly.
5.  **Commit your changes** with a clear and descriptive message.
6.  **Push your branch** to your forked repository.
7.  **Open a Pull Request** to the `main` branch of the original repository.

### Development Workflow

-   **Frontend:**
    -   Run `npm run dev` in the `frontend` directory.
    -   Changes will hot-reload.
-   **Backend:**
    -   Activate your Conda environment.
    -   Run `python backend/server.py`.
    -   You may need to restart the server for some changes to take effect.

### Code Style

-   **Frontend:** Adhere to standard Next.js/React best practices. Use TypeScript for type safety. Follow SHADcn and Tailwind CSS conventions. ESLint is configured for code linting.
-   **Backend:** Follow PEP 8 guidelines for Python code. Use clear variable names and add comments where necessary.

### Pull Request Process

1.  Ensure your branch is up-to-date with the `main` branch.
2.  Provide a clear title and detailed description for your Pull Request, explaining the changes and their purpose.
3.  Include screenshots or GIFs if your changes involve UI updates.
4.  Address any feedback or review comments promptly.
5.  Your PR will be reviewed and merged once approved.
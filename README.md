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
   conda activate [environment-name]
   ```

3. Start the backend server (refer to your backend documentation for specific commands).

## Project Structure

- `/frontend` - Next.js frontend application
  - `/app` - App router and pages
  - `/components` - Reusable React components
  - `/lib` - Utility functions and configuration

- `/backend` - Python backend application (details not fully visible in the provided code)

## Development

The project is organized using a standard Next.js structure with the app router. The main dashboard page renders components for file input and grouped images display.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
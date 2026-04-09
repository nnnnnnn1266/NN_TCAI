# TCAI Project Specification

## 1. Project Overview

TCAI (Turtle Care AI) is a domain-adapted question answering system based on Large Language Models (LLMs), designed for low-resource animal care scenarios.

The system focuses on improving knowledge alignment and response stability in specialized domains using high-quality datasets and parameter-efficient fine-tuning techniques.

---

## 2. Objectives

* Build a domain-specific QA system for turtle care
* Provide an API for model interaction
* Create a simple web interface for users
* Design a system that can be extended with RAG or multimodal models in the future

---

## 3. Core Methodology

### Model Concept

* Base model: LLaMA 3.1-8B (conceptual)
* Fine-tuning:

  * Supervised Fine-Tuning (SFT)
  * LoRA (Low-Rank Adaptation)
  * 4-bit Quantization

### Design Principles

* Focus on domain-specific knowledge alignment
* Use high-quality, manually validated dataset
* Do NOT use retrieval (RAG) in this version
* Keep experiment controlled (model-level evaluation only)

---

## 4. System Architecture

### Backend

* Framework: FastAPI
* Handles:

  * user input
  * API requests
  * calling model (or mock)

### Frontend

* Simple web page
* Input question → show answer

### Inference Module

* Start with mock response
* Later:

  * connect real model
  * add LoRA
  * add RAG

---

## 5. API Design

POST /ask

Request:
{
"question": "Why do turtles bask?"
}

Response:
{
"answer": "Turtles bask to regulate body temperature and absorb UV light."
}

---

## 6. Project Structure

TCAI-project/
│
├── app/
│   ├── main.py
│   ├── api/
│   ├── services/
│   └── models/
│
├── frontend/
│   └── index.html
│
├── data/
├── docs/
├── requirements.txt
├── README.md
└── .gitignore

---

## 7. Goal

Generate a working demo system:

* Backend API works
* Frontend can ask questions
* Ready for future LLM integration

---
title: Spam Email Detection
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: true
app_port: 8000
---

# 🛡️ Spam Email Detection — ML-Powered Gmail Protection

**A production-grade spam and phishing detection system with a Chrome extension, FastAPI backend, layered ML detection, explainable predictions, user feedback loop, retraining pipeline, and optional MySQL persistence.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-225%20passing-brightgreen.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-full%20production%20module%20coverage-success.svg)](#testing)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

Spam Email Detection is a complete spam and phishing detection platform that combines a Chrome extension for Gmail with a FastAPI backend running a layered detection engine. The system provides explainable predictions, captures user feedback, and supports retraining — making it suitable as both a real-world tool and a portfolio project demonstrating modern ML engineering practices.

### Why this project stands out

- **5-layer detection pipeline**: Whitelist → Trusted Service Catalog → Rule-Based Spam Detection → Benign Context Guard → Machine Learning Classification
- **Explainable AI**: Every prediction includes explanations showing which tokens and signals influenced the decision
- **Production-grade**: Docker deployment, env-based config, CORS protection, rate limiting, API key authentication, SHA-256 model integrity verification
- **PII redaction**: Emails, phone numbers, IPs, SSNs, and credit card numbers are automatically redacted at the API boundary
- **225 passing tests**: Full coverage of all production modules including integration tests that verify the real bootstrap flow with on-disk model artifacts

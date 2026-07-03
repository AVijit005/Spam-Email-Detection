# Known Issues

## Current Limitations

1. Checkpoint auto-resume for transformer training requires the `--resume` CLI flag. If training is interrupted without saving the checkpoint file, progress from that run is lost.
2. Gmail DOM selectors can change without notice, which may require updates in `extension/utils/domParser.js`. The extension has not been tested against every Gmail layout variant.
3. The ensemble requires the transformer model artifact (`transformer_model.pt`) for optimal accuracy. Without it, the system falls back to classical-only mode with reduced F1 on sophisticated phishing.
4. Retraining is user-triggered (`POST /v1/retrain`) rather than scheduled. There is no automatic retraining based on feedback volume or time intervals.
5. MySQL feedback storage is optional and env-driven. There is no in-app database credential management UI.
6. The API key authentication uses a single shared secret. Multi-user support with per-user API keys or JWT tokens is not yet implemented.
7. Kaggle GPU availability is subject to contention — training sessions may experience queue delays during peak hours.
8. Rate limiting uses in-memory storage (SlowAPI), which resets on server restart. In multi-worker Gunicorn deployments, limits are per-worker rather than global.

## Improvement Ideas

- Add scheduled retraining based on feedback volume threshold or cron schedule
- Implement JWT-based multi-user authentication
- Create Gmail regression fixtures for automated extension testing
- Add model A/B testing infrastructure with traffic splitting
- Build an admin dashboard for feedback review and model monitoring
- Distill DeBERTa-v3 to a smaller student model for faster inference
- Add CI/CD pipeline with automated testing and model evaluation
- Expand phishing phrase libraries for non-English languages

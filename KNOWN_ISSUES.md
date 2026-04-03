# Known Issues

## Current Limitations

1. The backend can now be deployed remotely, but it still does not implement authentication or multi-user access control.
2. Gmail DOM selectors can change, which may require updates in `extension/utils/domParser.js`.
3. The model is still limited by the quality and age of `backend/data/spam.csv`; feedback helps, but the base dataset is still relatively small for modern phishing patterns.
4. Retraining is automated and feedback-aware, but it is still user-triggered rather than scheduled.
5. MySQL feedback storage is optional and env-driven; there is no in-app database credential management yet.
6. Remote deployment is now supported, but the extension intentionally requires HTTPS for non-local backends.

## Improvement Ideas

- Add regression fixtures from real Gmail examples
- Add scheduled retraining or a reviewed-feedback queue before retraining
- Add authenticated or remote backend deployment support

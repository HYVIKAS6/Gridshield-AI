# Streamlit Community Cloud deployment

## Deployment coordinates

- Repository: `HYVIKAS6/Gridshield-AI`
- Branch: `main` (after merging the deployment-readiness pull request)
- Main file path: `app.py`
- Python: `3.11` (select in Advanced settings when creating the app)

Open https://share.streamlit.io/ and choose **Create app → Yup, I have an app**. Select the repository, branch and main file path above. Configure Python 3.11 before deployment.

## Pre-deployment validation

The GitHub Actions workflow `.github/workflows/streamlit-deployment-check.yml` installs dependencies, checks Python syntax, runs the tests, and verifies that Streamlit's health endpoint responds. **A passing CI run does not guarantee that every dashboard interaction works under Community Cloud's resource limits.**

## Important caveats

- Heavy packages (torch, pandapower, shap and xgboost) may require substantial memory and installation time.
- Trained model checkpoints in `models/saved/` are gitignored. Train models in the deployed session or arrange persistent external model storage if needed.
- Local filesystem writes are not a durable model registry on Community Cloud. Do not rely on files saved during one session surviving restarts.
- Check the mobile layout on a real phone and exercise training, inference, power-flow, upload and report features after deploying.
- If installation or runtime fails, open the app's **Manage app → Logs** panel and inspect the full traceback before changing dependencies.

## Local smoke test

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

## Rollback

If a change causes a regression, revert the deployment pull request on GitHub and let Community Cloud redeploy from the updated `main` branch.

# Civil Agent

Civil Agent is a structural engineering assistant for preliminary steel and concrete design studies.

## Local Run

```bash
cd beam_rl_project
streamlit run app.py
```

## Deployment Prep

This repo includes:

- [config.toml](/C:/Users/harsh/OneDrive/Desktop/Civil%20Agent,%20model%201/beam_rl_project/.streamlit/config.toml)
- [secrets.toml.example](/C:/Users/harsh/OneDrive/Desktop/Civil%20Agent,%20model%201/beam_rl_project/.streamlit/secrets.toml.example)

For Streamlit Cloud:

1. Push the repo to GitHub.
2. Add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in Streamlit secrets.
3. Deploy `app.py` as the entrypoint.

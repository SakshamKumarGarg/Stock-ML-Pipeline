# Web Frontend

This project includes an Express + HTML/CSS/JS dashboard in `web/`.

## Run Locally

```bash
cd web
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

The Express server calls `src/dashboard_api.py`, which reuses the existing processed CSVs and trained `.pkl` models to send JSON data to the browser UI.

## Structure

- `web/server.js`: Express server and API routes.
- `web/public/index.html`: Dashboard markup.
- `web/public/styles.css`: Responsive UI styling.
- `web/public/app.js`: Browser-side interactions and chart rendering.
- `src/dashboard_api.py`: Python adapter that evaluates saved models and returns JSON.

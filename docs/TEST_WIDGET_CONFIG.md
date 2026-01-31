# Testing widget config (DB + API + snippet)

Widget design is stored per bot in the DB. The embed script fetches config when it loads, so changing design in the dashboard updates the widget on the next page load.

## 1. Run backend and dashboard

**Terminal 1 – backend**
```bash
cd backend
# Set .env (DATABASE_URL, OPENAI_API_KEY, etc.)
python -m uvicorn api.app:create_app --factory --host 0.0.0.0 --port 5000
```

**Terminal 2 – dashboard (dev)**
```bash
cd dashboard
# Point at local backend (create .env or set inline)
# Windows PowerShell:
$env:VITE_API_BASE="http://localhost:5000"; npm run dev
# Or add to dashboard/.env: VITE_API_BASE=http://localhost:5000
npm run dev
```

Open the dashboard (e.g. http://localhost:5173), log in if required.

---

## 2. Test create-bot flow (save config to DB)

1. **Create bot** – Go to Create bot → Name + Website → enter name and URL → Next.
2. **Select URLs** – Select URLs → (optional) pick some → Start training.
3. **Training** – Wait until you can continue (or use auto-advance to Design).
4. **Design (step 4)** – Change:
   - Position: Bottom left
   - Accent color: e.g. `#ffae00`
   - Widget title: e.g. `Support`
   - Placeholder: e.g. `Ask us anything...`
   - Theme: Light or Dark
5. Click **Continue** – Config is saved to the DB for this bot (PUT `/v1/org/bots/{bot_id}/widget-config`).
6. **Embed (step 5)** – Copy the snippet. It should look like:
   ```html
   <script async src="http://localhost:5000/widget/widget.js" data-bot-key="pk_..." data-api-base="http://localhost:5000"></script>
   ```
   No design options in the tag; design comes from the API.

---

## 3. Test widget on a page (fetch config from API)

1. Create a minimal HTML file (e.g. `test-widget.html`) on your machine:
   ```html
   <!DOCTYPE html>
   <html>
   <head><title>Widget test</title></head>
   <body>
     <h1>Widget test</h1>
     <p>You should see the chat widget (design from DB).</p>
     <!-- Paste the snippet you copied (use your pk_... and API base) -->
     <script async src="http://localhost:5000/widget/widget.js" data-bot-key="pk_YOUR_KEY" data-api-base="http://localhost:5000"></script>
   </body>
   </html>
   ```
2. Replace `pk_YOUR_KEY` with the real publishable key from the Embed step.
3. Open the file in a browser (e.g. `file:///path/to/test-widget.html`).
   - **CORS**: If the backend only allows verified domains, the widget’s fetch to `GET /v1/pk/{pk}/widget-config` may be blocked when opening from `file://` or another origin. To test without domain verification, you can temporarily allow all origins for that endpoint, or serve the HTML from the same origin as the API (e.g. a simple route that returns this HTML from the backend).
4. You should see the chat widget with the design you set (position, color, title, placeholder, theme). That proves:
   - `widget.js` called `GET /v1/pk/{pk}/widget-config`
   - Backend returned the saved config
   - Iframe was built from that config

---

## 4. Test “change later” (update config, reload page)

1. In the dashboard, go to **Bots** → open the bot you created.
2. If you have a **Design** tab that loads/saves widget config: change e.g. accent color or title → Save.
   - If there is no Design tab yet, use the API directly (see below).
3. Reload the test HTML page. The widget should show the **new** design (same snippet, config refetched from DB).

**Optional – change config via API (no Design tab)**

```bash
# Get your bot_id from the dashboard (e.g. from the URL when viewing the bot).
# You need a valid auth token (e.g. from browser DevTools → Application → copy token after login).

curl -X PUT "http://localhost:5000/v1/org/bots/BOT_ID/widget-config" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT" \
  -d '{"position":"bottom-left","color":"#00aa00","title":"Help","size":"medium","placeholder":"Ask..."}'
```

Then reload the test page; the widget should show the new color/title.

---

## 5. Quick API checks (no UI)

**GET widget-config (public, no auth)**  
Replace `pk_xxx` with a real publishable key.

```bash
curl -s "http://localhost:5000/v1/pk/pk_xxx/widget-config"
```

- If the bot has saved config: you get JSON (e.g. `{"position":"bottom-left","color":"#ffae00",...}`).
- If none: you get `{}`.

**DB**  
If you use Postgres, you can confirm the column and value:

```sql
SELECT bot_id, display_name, LEFT(widget_config, 80) AS config_preview FROM bots LIMIT 5;
```

---

## 6. Checklist

- [ ] Backend runs; dashboard runs and points at backend.
- [ ] Create bot → Design → change options → Continue (config saved).
- [ ] Embed step shows minimal snippet (only `data-bot-key` and `data-api-base`).
- [ ] Test HTML with that snippet loads widget and shows the saved design.
- [ ] After updating config (Design tab or PUT API), reloading the test page shows the new design.

If the widget does not appear or shows default look, check:

- Browser console for CORS or network errors (e.g. GET widget-config blocked).
- Backend logs for 404/500 on `/v1/pk/.../widget-config`.
- That the bot’s `widget_config` in the DB is non-empty after saving from Design.

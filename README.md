# Caffeine Critique

A small Flask site built on my coffee shop reviews. Describe a coffee (price, roast,
espresso quality, and so on) and it predicts how I'd rate it out of 100, with an
uncertainty estimate. It also lists every shop I've rated, filterable by city.

## How it works

Each review is a markdown note in `Coffee Shops/` with YAML front matter:

```yaml
---
rating: 88.5
location: Austin, Texas
price: Average            # Cheap | Average | Expensive
roast level: Medium       # Light | Medium | Medium-dark | Dark
espresso: Good            # Horrible | Mediocre | Good | Exceptional
sweetness: Balanced       # Subtle | Sweet | Balanced
strength: Balanced        # Weak | Strong | Balanced
bitter: No                # Yes | No (same for the fields below)
house syrups: Yes
specialty drinks: Yes
espresso variety: No
mixed: Yes
edible decor: No
---
```

On startup the app reads every note and trains a Gaussian Process
([BoTorch](https://botorch.org/) `MixedSingleTaskGP`) on all of them. The form's
prediction is the model's mean rating, plus or minus one standard deviation.

To add a review, drop a new note into `Coffee Shops/` and restart the app.

## Run locally

Requires Python 3.10 (see `runtime.txt`).

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8080.

## Deploy

The `Procfile` runs the app with gunicorn (`web: gunicorn app:app`).

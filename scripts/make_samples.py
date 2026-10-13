"""Generate the bundled sample images (fictional data) into samples/.

Run with:  python scripts/make_samples.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "samples"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def sales_chart() -> Image.Image:
    revenue = {"Q1": 42, "Q2": 58, "Q3": 37, "Q4": 71}  # thousands of dollars
    img = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(img)
    d.text((450, 40), "Harbor Street Coffee: 2025 revenue by quarter", font=font(28), fill="#1f2933", anchor="mm")
    d.text((450, 78), "Thousands of US dollars", font=font(18), fill="#616e7c", anchor="mm")

    left, base, top, scale = 110, 500, 120, 5  # px per $1k
    for value in range(0, 81, 20):
        y = base - value * scale
        d.line([(left, y), (840, y)], fill="#e4e7eb", width=1)
        d.text((left - 14, y), str(value), font=font(18), fill="#616e7c", anchor="rm")
    d.line([(left, top), (left, base), (840, base)], fill="#1f2933", width=2)

    for i, (quarter, value) in enumerate(revenue.items()):
        x = left + 60 + i * 175
        y = base - value * scale
        d.rectangle([(x, y), (x + 100, base - 1)], fill="#c0392b" if value < 40 else "#2b6cb0")
        d.text((x + 50, y - 16), f"${value}k", font=font(20), fill="#1f2933", anchor="mm")
        d.text((x + 50, base + 24), quarter, font=font(20), fill="#1f2933", anchor="mm")
    d.text((450, 565), "Q3 dip: main location closed three weeks for renovation", font=font(17), fill="#c0392b", anchor="mm")
    return img


def receipt() -> Image.Image:
    items = [("Flat white", 2, 4.50), ("Almond croissant", 1, 3.75), ("Avocado toast", 1, 9.20), ("Cold brew (L)", 1, 5.00)]
    subtotal = sum(qty * price for _, qty, price in items)
    tax = round(subtotal * 0.08, 2)

    img = Image.new("RGB", (480, 720), "#fbfaf5")
    d = ImageDraw.Draw(img)
    d.text((240, 50), "HARBOR STREET COFFEE", font=font(26), fill="black", anchor="mm")
    d.text((240, 84), "118 Harbor St, Portland", font=font(17), fill="#444", anchor="mm")
    d.text((240, 110), "14 Mar 2025  09:42   Order #2087", font=font(17), fill="#444", anchor="mm")
    d.line([(30, 140), (450, 140)], fill="black", width=2)

    y = 170
    for name, qty, price in items:
        d.text((30, y), f"{qty} x {name}", font=font(20), fill="black")
        d.text((450, y), f"{qty * price:.2f}", font=font(20), fill="black", anchor="ra")
        y += 44
    d.line([(30, y + 6), (450, y + 6)], fill="black", width=1)

    y += 30
    for label, amount, size in [("Subtotal", subtotal, 20), ("Tax (8%)", tax, 20), ("TOTAL", subtotal + tax, 26)]:
        d.text((30, y), label, font=font(size), fill="black")
        d.text((450, y), f"${amount:.2f}", font=font(size), fill="black", anchor="ra")
        y += 46
    d.text((240, y + 40), "Paid by card  **** 0000", font=font(17), fill="#444", anchor="mm")
    d.text((240, y + 90), "Thank you! See you tomorrow.", font=font(19), fill="black", anchor="mm")
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sales_chart().save(OUT / "sales_chart.png")
    receipt().save(OUT / "cafe_receipt.png")
    print(f"Wrote samples to {OUT}")

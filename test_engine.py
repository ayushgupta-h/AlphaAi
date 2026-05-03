from engine import generate_prediction
from data_fetcher import fetch_current_price, fetch_binance_data

print('🔥 Testing Core Engine...')
current_price = fetch_current_price()
print(f'💰 Current BTC Price: ${current_price:,.2f}')

df = fetch_binance_data(limit=50)
historical_prices = df['close'].tolist()
print(f'📈 Historical data: {len(historical_prices)} hourly prices')

prediction = generate_prediction(
    current_price=current_price,
    historical_prices=historical_prices,
    confidence_level=0.95
)

print('🎯 PREDICTION RESULTS:')
print(f'   Current Price: ${prediction["current_price"]:,.2f}')
print(f'   95% Interval: ${prediction["lower_bound"]:,.2f} - ${prediction["upper_bound"]:,.2f}')
print(f'   Interval Width: ${prediction["interval_width"]:,.2f}')
print(f'   Volatility: {prediction["volatility"]:.4f}')
print(f'   Drift: {prediction["drift"]:.4f}')
print('✅ Core engine working perfectly!')
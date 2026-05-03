// MongoDB initialization script for Bitcoin Prediction API
// This script sets up the database schema and indexes

// Switch to the bitcoin_predictions database
db = db.getSiblingDB('bitcoin_predictions');

// Create the predictions collection with schema validation
db.createCollection('predictions', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['timestamp', 'current_price', 'lower_bound', 'upper_bound', 'confidence_level'],
      properties: {
        timestamp: {
          bsonType: 'date',
          description: 'Prediction generation timestamp - required'
        },
        current_price: {
          bsonType: 'number',
          minimum: 0,
          description: 'Current Bitcoin price in USD - required and must be positive'
        },
        lower_bound: {
          bsonType: 'number',
          minimum: 0,
          description: 'Lower bound of prediction interval - required and must be positive'
        },
        upper_bound: {
          bsonType: 'number',
          minimum: 0,
          description: 'Upper bound of prediction interval - required and must be positive'
        },
        confidence_level: {
          bsonType: 'number',
          minimum: 0.5,
          maximum: 0.999,
          description: 'Confidence level - required and must be between 0.5 and 0.999'
        },
        prediction_horizon: {
          bsonType: 'number',
          minimum: 0,
          description: 'Prediction horizon in hours'
        },
        volatility: {
          bsonType: 'number',
          minimum: 0,
          description: 'EWMA volatility estimate'
        },
        drift: {
          bsonType: 'number',
          description: 'Estimated drift parameter'
        },
        model_version: {
          bsonType: 'string',
          description: 'Mathematical model version'
        },
        created_at: {
          bsonType: 'date',
          description: 'Document creation timestamp'
        }
      }
    }
  }
});

// Create indexes for efficient querying
// Compound index for date range queries with confidence level
db.predictions.createIndex(
  { timestamp: -1, confidence_level: 1 },
  { name: 'timestamp_confidence_idx' }
);

// Single field index for latest prediction queries
db.predictions.createIndex(
  { timestamp: -1 },
  { name: 'timestamp_desc_idx' }
);

// Index for created_at field (useful for TTL if needed)
db.predictions.createIndex(
  { created_at: 1 },
  { name: 'created_at_idx' }
);

// Optional: Create TTL index for automatic data retention (90 days)
// Uncomment the following line if you want automatic data expiration
// db.predictions.createIndex(
//   { created_at: 1 },
//   { expireAfterSeconds: 7776000, name: 'ttl_idx' }  // 90 days
// );

print('MongoDB initialization completed successfully');
print('Created predictions collection with schema validation');
print('Created indexes for efficient querying');
print('Database is ready for Bitcoin Prediction API');
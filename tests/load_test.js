import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

// ✅ Custom metric to track latency
let latency = new Trend('inference_latency');

// ✅ Define test configuration
export let options = {
  stages: [
    { duration: '30s', target: 50 },  // ramp up to 50 users
    { duration: '1m', target: 50 },   // stay at 50
    { duration: '30s', target: 0 },   // ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests < 500ms
    http_req_failed: ['rate<0.05'],    // < 5% errors
  },
};

// ✅ Define API endpoint
const BASE_URL = 'https://carprice-api-375680962785-us-east1.run.app/predict';

export default function () {
  const payload = JSON.stringify({
    year: 2019,
    odometer: 42000,
    condition: 'Good',
    fuel: 'Gas',
    transmission: 'Automatic',
    manufacturer: 'Toyota',
    model: 'highlander'
  });

  const headers = { 'Content-Type': 'application/json' };
  const res = http.post(BASE_URL, payload, { headers });

  // Track latency
  latency.add(res.timings.duration);

  // Validate the response
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1); // small pause between requests
}

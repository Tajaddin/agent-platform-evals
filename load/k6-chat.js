// k6 load test for the chat API.
//
//   docker compose --profile load run --rm k6                # against the compose stack (fake provider)
//   k6 run -e BASE_URL=http://localhost:8000 -e API_KEY=... load/k6-chat.js
//
// With AGENT_PROVIDER=fake this measures the pipeline (API, graph, guard, DuckDB, memory) with zero model
// latency. It says nothing about model speed; with a live provider the same script measures the full path.
// Questions come from the golden set so the fake planner recognises them. Raise AGENT_RATE_LIMIT on the API
// before running, otherwise the limiter answers 429 after 30 requests per key.

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Counter } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://api:8000";
const API_KEY = __ENV.API_KEY || "change-me-local-key";
const VUS = Number(__ENV.VUS || 10);
const DURATION = __ENV.DURATION || "60s";

const QUESTIONS = [
  "How many weather stations reported on 2026-10-10?",
  "What was the highest maximum temperature recorded in Florida on 2026-10-10?",
  "Which state had the highest average maximum temperature on 2026-10-12?",
  "How many stations are in Florida?",
  "What was the average daily precipitation across Texas stations on 2026-10-05?",
  "Which station recorded the coldest temperature on 2026-10-14?",
  "Show the daily average maximum temperature in Florida for each day.",
  "Which Florida station had the most rain on 2026-10-09?",
  "Which states had snow on 2026-10-13?",
  "How many stations reported on every day of the period?",
  "What was the average wind speed across all stations on 2026-10-06?",
  "Which states reported fewer than 10 stations on 2026-10-14?",
  "What was the national average maximum temperature on 2026-10-10?",
  "How many stations have a WMO id?",
  "Hello",
  "Show me the weather",
];

const turnLatency = new Trend("agent_turn_latency_ms", true);
const answered = new Counter("agent_answered");
const nonAnswered = new Counter("agent_not_answered");

export const options = {
  scenarios: {
    chat: {
      executor: "constant-vus",
      vus: VUS,
      duration: DURATION,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{endpoint:chat}": ["p(95)<1000"],
    checks: ["rate>0.99"],
  },
};

export default function () {
  const thread = `k6-${__VU}-${__ITER}`;
  const question = QUESTIONS[(__VU * 7 + __ITER) % QUESTIONS.length];
  const res = http.post(
    `${BASE_URL}/v1/chat`,
    JSON.stringify({ thread_id: thread, message: question }),
    { headers: { "Content-Type": "application/json", "X-API-Key": API_KEY }, tags: { endpoint: "chat" } },
  );
  const ok = check(res, {
    "status 200": (r) => r.status === 200,
    "has status field": (r) => r.status === 200 && typeof r.json("status") === "string",
  });
  if (ok) {
    const body = res.json();
    turnLatency.add(body.latency_ms);
    if (body.status === "answered") answered.add(1);
    else nonAnswered.add(1);
  }
  sleep(0.1);
}

export function handleSummary(data) {
  const m = data.metrics;
  const pick = (name, stat) => (m[name] && m[name].values && m[name].values[stat] !== undefined ? m[name].values[stat] : null);
  const summary = {
    vus: VUS,
    duration: DURATION,
    requests: pick("http_reqs", "count"),
    requests_per_second: pick("http_reqs", "rate"),
    failed_rate: pick("http_req_failed", "rate"),
    http_p50_ms: pick("http_req_duration{endpoint:chat}", "med"),
    http_p95_ms: pick("http_req_duration{endpoint:chat}", "p(95)"),
    http_p99_ms: pick("http_req_duration{endpoint:chat}", "p(99)"),
    agent_turn_p50_ms: pick("agent_turn_latency_ms", "med"),
    agent_turn_p95_ms: pick("agent_turn_latency_ms", "p(95)"),
    answered: pick("agent_answered", "count"),
    not_answered: pick("agent_not_answered", "count"),
    thresholds_passed: Object.values(data.metrics).every((x) => !x.thresholds || Object.values(x.thresholds).every((t) => t.ok)),
  };
  return {
    stdout: JSON.stringify(summary, null, 2) + "\n",
    "/results/summary.json": JSON.stringify(summary, null, 2) + "\n",
  };
}

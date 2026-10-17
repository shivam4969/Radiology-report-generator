// SQLite storage for reads and ratings (better-sqlite3).
//
// reads    one row per analysis (UI or bench); holds the model, the report
//          text, its hash and the reading path
// ratings  one row per human verdict on a read; the benchmark joins the two

import Database from 'better-sqlite3';
import { createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const DB_PATH = process.env.DB_PATH || resolve(process.cwd(), 'data/openread.db');

let db;
export function getDb(path = DB_PATH) {
  if (db) return db;
  mkdirSync(dirname(path), { recursive: true });
  db = new Database(path);
  db.pragma('journal_mode = WAL');
  db.exec(`
    CREATE TABLE IF NOT EXISTS reads (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      engine TEXT NOT NULL,
      model TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'ui',      -- ui | bench
      image_name TEXT,
      image_hash TEXT,
      history TEXT,
      report_hash TEXT NOT NULL,
      report_json TEXT NOT NULL,
      path_json TEXT,
      duration_ms INTEGER
    );
    CREATE TABLE IF NOT EXISTS ratings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      read_id INTEGER REFERENCES reads(id),
      engine TEXT NOT NULL,
      model TEXT NOT NULL,
      report_hash TEXT NOT NULL,
      acceptable INTEGER NOT NULL,            -- 1 acceptable, 0 not acceptable
      reason TEXT,
      study_type TEXT,                        -- e.g. chest-1-view, chest-2-view
      icu INTEGER,                            -- 1 if portable/ICU
      modality TEXT                           -- e.g. XR, CT
    );
    CREATE INDEX IF NOT EXISTS idx_ratings_model ON ratings(model, study_type);
    CREATE INDEX IF NOT EXISTS idx_reads_hash ON reads(report_hash);
  `);
  return db;
}

export function hashReport(report) {
  const text = typeof report === 'string' ? report : `${report?.exam ?? ''}\n${report?.findings ?? ''}\n${report?.impression ?? ''}`;
  return createHash('sha256').update(text.trim()).digest('hex').slice(0, 16);
}

export function hashBytes(buf) {
  return createHash('sha256').update(buf).digest('hex').slice(0, 16);
}

export function insertRead(row) {
  const d = getDb();
  const stmt = d.prepare(`INSERT INTO reads (engine, model, source, image_name, image_hash, history, report_hash, report_json, path_json, duration_ms)
    VALUES (@engine, @model, @source, @image_name, @image_hash, @history, @report_hash, @report_json, @path_json, @duration_ms)`);
  const info = stmt.run({
    engine: row.engine,
    model: row.model,
    source: row.source || 'ui',
    image_name: row.imageName || null,
    image_hash: row.imageHash || null,
    history: row.history || null,
    report_hash: row.reportHash,
    report_json: JSON.stringify(row.report),
    path_json: JSON.stringify(row.path || []),
    duration_ms: row.durationMs ?? null,
  });
  return Number(info.lastInsertRowid);
}

export function insertRating(row) {
  const d = getDb();
  const stmt = d.prepare(`INSERT INTO ratings (read_id, engine, model, report_hash, acceptable, reason, study_type, icu, modality)
    VALUES (@read_id, @engine, @model, @report_hash, @acceptable, @reason, @study_type, @icu, @modality)`);
  const info = stmt.run({
    read_id: row.readId ?? null,
    engine: row.engine,
    model: row.model,
    report_hash: row.reportHash,
    acceptable: row.acceptable ? 1 : 0,
    reason: row.reason || null,
    study_type: row.studyType || null,
    icu: row.icu == null ? null : row.icu ? 1 : 0,
    modality: row.modality || null,
  });
  return Number(info.lastInsertRowid);
}

/** Acceptable rate per model per study type. */
export function benchmarkSummary() {
  const d = getDb();
  const rows = d
    .prepare(
      `SELECT model, COALESCE(study_type,'unspecified') AS study_type,
              COUNT(*) AS rated, SUM(acceptable) AS acceptable,
              ROUND(100.0 * SUM(acceptable) / COUNT(*), 1) AS acceptable_pct
       FROM ratings GROUP BY model, study_type ORDER BY model, study_type`,
    )
    .all();
  const reads = d.prepare(`SELECT model, COUNT(*) AS reads, SUM(source='bench') AS bench_reads FROM reads GROUP BY model`).all();
  const unrated = d
    .prepare(
      `SELECT r.model, COUNT(*) AS unrated FROM reads r
       LEFT JOIN ratings g ON g.read_id = r.id WHERE g.id IS NULL GROUP BY r.model`,
    )
    .all();
  return { ratings: rows, reads, unrated };
}

/** Flat join of reads and ratings for CSV export. */
export function benchmarkRows() {
  const d = getDb();
  return d
    .prepare(
      `SELECT r.id AS read_id, r.created_at AS read_at, r.engine, r.model, r.source, r.image_name, r.image_hash,
              r.report_hash, r.duration_ms,
              g.id AS rating_id, g.created_at AS rated_at, g.acceptable, g.reason, g.study_type, g.icu, g.modality
       FROM reads r LEFT JOIN ratings g ON g.read_id = r.id ORDER BY r.id`,
    )
    .all();
}

/** Oldest unrated read, for a "rate the bench results" flow. */
export function nextUnrated() {
  const d = getDb();
  return d
    .prepare(
      `SELECT r.* FROM reads r LEFT JOIN ratings g ON g.read_id = r.id WHERE g.id IS NULL ORDER BY r.id LIMIT 1`,
    )
    .get();
}

export function toCsv(rows) {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  const esc = (v) => {
    if (v == null) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n') + '\n';
}

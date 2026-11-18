-- Star schema for half-hourly NEM market data. The source view is registered by Python as `src`.
CREATE OR REPLACE TABLE dim_region AS
SELECT row_number() OVER (ORDER BY region)::INTEGER AS region_key,
       region AS region_code,
       CASE region
           WHEN 'NSW1' THEN 'New South Wales'
           WHEN 'QLD1' THEN 'Queensland'
           WHEN 'SA1' THEN 'South Australia'
           WHEN 'TAS1' THEN 'Tasmania'
           WHEN 'VIC1' THEN 'Victoria'
       END AS region_name
FROM (SELECT DISTINCT region FROM src);

CREATE OR REPLACE TABLE dim_time AS
SELECT strftime(ts, '%Y%m%d%H%M')::BIGINT AS time_key,
       ts AS period_end,
       CAST(ts - INTERVAL 30 MINUTE AS DATE) AS trading_day,
       ((hour(ts - INTERVAL 30 MINUTE) * 2) + (minute(ts - INTERVAL 30 MINUTE) / 30))::INTEGER + 1
           AS trading_period,
       year(ts - INTERVAL 30 MINUTE) AS year,
       quarter(ts - INTERVAL 30 MINUTE) AS quarter,
       month(ts - INTERVAL 30 MINUTE) AS month,
       isodow(ts - INTERVAL 30 MINUTE) AS iso_day_of_week,
       isodow(ts - INTERVAL 30 MINUTE) >= 6 AS is_weekend
FROM (SELECT DISTINCT ts FROM src);

CREATE OR REPLACE TABLE fact_market_interval AS
SELECT t.time_key, r.region_key, s.price AS price_aud_per_mwh, s.demand_mw, s.n_obs
FROM src AS s
JOIN dim_time AS t ON s.ts = t.period_end
JOIN dim_region AS r ON s.region = r.region_code;

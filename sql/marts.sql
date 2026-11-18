-- Analytical marts built only on the star schema.
CREATE OR REPLACE VIEW mart_daily_market AS
WITH intervals AS (
    SELECT r.region_code, t.trading_day, t.trading_period, f.price_aud_per_mwh AS price, f.demand_mw,
           row_number() OVER w_asc AS rank_low,
           row_number() OVER w_desc AS rank_high
    FROM fact_market_interval AS f
    JOIN dim_time AS t USING (time_key)
    JOIN dim_region AS r USING (region_key)
    WINDOW w_asc AS (PARTITION BY r.region_code, t.trading_day ORDER BY f.price_aud_per_mwh),
           w_desc AS (PARTITION BY r.region_code, t.trading_day ORDER BY f.price_aud_per_mwh DESC)
)
SELECT region_code,
       trading_day,
       count(*) AS periods,
       avg(price) AS mean_price,
       sum(price * demand_mw) / nullif(sum(demand_mw), 0) AS demand_weighted_price,
       avg(price) FILTER (WHERE rank_high <= 4) - avg(price) FILTER (WHERE rank_low <= 4)
           AS top2h_bottom2h_spread,
       avg(CASE WHEN price < 0 THEN 1.0 ELSE 0.0 END) AS negative_price_share,
       max(price) AS max_price,
       avg(demand_mw) AS mean_demand_mw
FROM intervals
GROUP BY region_code, trading_day;

CREATE OR REPLACE VIEW mart_monthly_market AS
SELECT region_code,
       date_trunc('month', trading_day) AS month,
       avg(mean_price) AS mean_price,
       avg(demand_weighted_price) AS demand_weighted_price,
       avg(top2h_bottom2h_spread) AS mean_daily_spread,
       avg(negative_price_share) AS negative_price_share,
       max(max_price) AS max_price
FROM mart_daily_market
WHERE periods = 48
GROUP BY region_code, date_trunc('month', trading_day);

CREATE OR REPLACE VIEW mart_spread_trend AS
SELECT *,
       avg(mean_daily_spread) OVER (
           PARTITION BY region_code ORDER BY month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
       ) AS spread_12m_average,
       mean_daily_spread / nullif(lag(mean_daily_spread, 12) OVER (
           PARTITION BY region_code ORDER BY month
       ), 0) - 1 AS spread_yoy_change
FROM mart_monthly_market;

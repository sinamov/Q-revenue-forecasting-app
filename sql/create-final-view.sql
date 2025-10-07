-- This section safely deletes old objects in the correct order.

IF EXISTS (SELECT * FROM sys.views WHERE name = 'v_financials_quarterly') DROP VIEW v_financials_quarterly;
GO
IF EXISTS (SELECT * FROM sys.external_tables WHERE name = 'fct_financials_quarterly') DROP EXTERNAL TABLE fct_financials_quarterly;
GO
IF EXISTS (SELECT * FROM sys.external_tables WHERE name = 'stg_raw_aapl') DROP EXTERNAL TABLE stg_raw_aapl;
GO
IF EXISTS (SELECT * FROM sys.external_tables WHERE name = 'stg_raw_msft') DROP EXTERNAL TABLE stg_raw_msft;
GO
IF EXISTS (SELECT * FROM sys.external_tables WHERE name = 'stg_raw_googl') DROP EXTERNAL TABLE stg_raw_googl;
GO
IF EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'ParquetFormat') DROP EXTERNAL FILE FORMAT ParquetFormat;
GO
IF EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'raw_data') DROP EXTERNAL DATA SOURCE raw_data;
GO
IF EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'financials_data') DROP EXTERNAL DATA SOURCE financials_data;
GO
IF EXISTS (SELECT * FROM sys.database_scoped_credentials WHERE name = 'msi_cred') DROP DATABASE SCOPED CREDENTIAL msi_cred;
GO
IF EXISTS (SELECT * FROM sys.symmetric_keys WHERE name = '##MS_DatabaseMasterKey##') DROP MASTER KEY;
GO

-- == Recreate All Objects Cleanly ==

CREATE MASTER KEY ENCRYPTION BY PASSWORD = '2.Movahedi';
GO
CREATE DATABASE SCOPED CREDENTIAL msi_cred WITH IDENTITY = 'Managed Identity';
GO

CREATE EXTERNAL DATA SOURCE raw_data WITH ( LOCATION = 'https://revforecastersma.dfs.core.windows.net/raw/' , CREDENTIAL = msi_cred);
GO
CREATE EXTERNAL DATA SOURCE financials_data WITH ( LOCATION = 'https://revforecastersma.dfs.core.windows.net/processed/' , CREDENTIAL = msi_cred);
GO
CREATE EXTERNAL FILE FORMAT ParquetFormat WITH (FORMAT_TYPE = PARQUET);
GO

-- Create raw tables
CREATE EXTERNAL TABLE stg_raw_aapl ( report_date VARCHAR(20), loaded_at BIGINT, total_revenue BIGINT, research_and_development BIGINT, selling_general_and_administrative BIGINT, net_income BIGINT, total_assets BIGINT, total_liabilities BIGINT, operating_cashflow BIGINT, capital_expenditures BIGINT ) WITH ( LOCATION = 'AAPL_financials.parquet', DATA_SOURCE = raw_data, FILE_FORMAT = ParquetFormat );
GO
CREATE EXTERNAL TABLE stg_raw_msft ( report_date VARCHAR(20), loaded_at BIGINT, total_revenue BIGINT, research_and_development BIGINT, selling_general_and_administrative BIGINT, net_income BIGINT, total_assets BIGINT, total_liabilities BIGINT, operating_cashflow BIGINT, capital_expenditures BIGINT ) WITH ( LOCATION = 'MSFT_financials.parquet', DATA_SOURCE = raw_data, FILE_FORMAT = ParquetFormat );
GO
CREATE EXTERNAL TABLE stg_raw_googl ( report_date VARCHAR(20), loaded_at BIGINT, total_revenue BIGINT, research_and_development BIGINT, selling_general_and_administrative BIGINT, net_income BIGINT, total_assets BIGINT, total_liabilities BIGINT, operating_cashflow BIGINT, capital_expenditures BIGINT ) WITH ( LOCATION = 'GOOGL_financials.parquet', DATA_SOURCE = raw_data, FILE_FORMAT = ParquetFormat );
GO

-- Run CETAS to transform the data
CREATE EXTERNAL TABLE fct_financials_quarterly
WITH ( LOCATION = 'fct_financials_quarterly/', DATA_SOURCE = financials_data, FILE_FORMAT = ParquetFormat )
AS
WITH RawFinancialsCombined AS (
    SELECT *, 'AAPL' AS ticker FROM stg_raw_aapl UNION ALL
    SELECT *, 'MSFT' AS ticker FROM stg_raw_msft UNION ALL
    SELECT *, 'GOOGL' AS ticker FROM stg_raw_googl
),
CleanFinancials AS (
    SELECT
        CAST(report_date AS DATE) AS report_date,
        DATEADD(month, 3, DATEFROMPARTS(YEAR(CAST(report_date AS DATE)), ((MONTH(CAST(report_date AS DATE)) - 1) / 3) * 3 + 1, 1)) AS prediction_quarter,
        ticker, total_revenue AS revenues, research_and_development AS research_and_development_expense,
        selling_general_and_administrative AS selling_general_and_administrative_expense,
        net_income, total_assets AS assets, total_liabilities AS liabilities
    FROM RawFinancialsCombined
),
CleanMacro AS (
    SELECT
        DATEADD(month, 3, CAST(report_date AS DATE)) AS prediction_quarter,
        gdp, cpi, unemployment
    FROM OPENROWSET( BULK 'stg_fred_macro_quarterly.parquet', DATA_SOURCE = 'financials_data', FORMAT = 'PARQUET' ) AS r
)
SELECT
    fin.prediction_quarter, fin.ticker, fin.report_date, fin.revenues, fin.research_and_development_expense,
    fin.selling_general_and_administrative_expense, fin.net_income, fin.assets, fin.liabilities,
    macro.gdp, macro.cpi, macro.unemployment
FROM CleanFinancials fin LEFT JOIN CleanMacro macro ON fin.prediction_quarter = macro.prediction_quarter;
GO

-- Create the final view
CREATE OR ALTER VIEW v_financials_quarterly AS SELECT * FROM fct_financials_quarterly;
GO
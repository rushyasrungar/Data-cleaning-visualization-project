# Data Cleaning Report

| Metric | Before | After |
|---|---:|---:|
| Rows | 4,278 | 4,114 |
| Exact duplicate rows | 164 | 0 |
| Duplicate order IDs | 164 | 0 |
| Missing cells | 1,548 | 0 |

## Step by step

1. **Standardise text** - trimmed whitespace, unified case, mapped abbreviations (unique labels before -> after: {'region': 19, 'category': 17, 'payment_method': 16} -> {'region': 5, 'category': 5, 'payment_method': 5}).
2. **Fix data types** - parsed 4 mixed date formats (0 unparseable); stripped currency symbols from 348 `sales` values.
3. **Remove duplicates** - 164 exact duplicates and 0 repeated order IDs removed.
4. **Invalid values & outliers** - customer age out of range: 15, delivery days out of range: 14, discount out of range: 0, rating out of range: 0, quantity out of range: 25, unit price extreme: 20.
5. **Missing values** - customer_name: 90, customer_age: 290, region: 163, category: 25, product: 40, quantity: 25, discount: 199, payment_method: 116, delivery_days: 194, rating: 401.
6. **Sales recomputed** from quantity x price x (1 - discount) for 156 rows.

## IQR outlier counts (1.5 x IQR rule)

| Column | Before | After |
|---|---:|---:|
| quantity | 26 | 0 |
| unit_price | 77 | 75 |
| customer_age | 24 | 33 |
| delivery_days | 17 | 2 |
| sales | 309 | 293 |

> Some IQR-flagged rows remain **on purpose**. `sales` (e.g. 5 x smartwatch), `unit_price` (prices are pooled across very different categories) and `customer_age` (natural tails) contain genuine values. Errors were caught with domain rules (valid ranges, price vs. product median), and the plain IQR rule is reported for transparency instead of blindly capping real data.

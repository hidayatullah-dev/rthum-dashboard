# 🧮 Formula Technical Documentation

## Mathematical Foundations

### 1. Scoring Systems

#### Linear Scoring
**Formula**: `Score`
**Purpose**: Direct use of raw score values
**Range**: 0-100 (typically)
**Use Case**: When raw scores are already normalized

#### Exponential Scoring
**Formula**: `Score ** 2`
**Purpose**: Emphasizes higher scores exponentially
**Range**: 0-10,000 (for scores 0-100)
**Use Case**: When you want to heavily weight top performers

#### Logarithmic Scoring
**Formula**: `Score.apply(lambda x: math.log(x + 1) if x > 0 else 0)`
**Purpose**: Compresses high scores, expands low scores
**Range**: 0-4.61 (for scores 0-100)
**Use Case**: When you want to reduce the impact of extreme high scores

#### Square Root Scoring
**Formula**: `Score ** 0.5`
**Purpose**: Moderate emphasis on higher scores
**Range**: 0-10 (for scores 0-100)
**Use Case**: Balanced approach between linear and exponential

### 2. Normalization Techniques

#### Min-Max Normalization
**Formula**: `(Score - Score.min()) / (Score.max() - Score.min())`
**Purpose**: Scales values to 0-1 range
**Range**: [0, 1]
**Use Case**: When you need standardized values for comparison

#### Z-Score Standardization
**Formula**: `(Score - Score.mean()) / Score.std()`
**Purpose**: Centers data around 0 with unit variance
**Range**: Approximately [-3, 3] (99.7% of data)
**Use Case**: When you need to identify outliers and compare across different scales

#### Percentile Ranking
**Formula**: `Score.rank(pct=True) * 100`
**Purpose**: Converts scores to percentile ranks
**Range**: [0, 100]
**Use Case**: When you want to know relative position in distribution

### 3. Composite Scoring Systems

#### Weighted Linear Combination
**Formula**: `Score * w1 + (Amount spent / 1000) * w2 + Proposals * w3`
**Where**: `w1 + w2 + w3 = 1`
**Purpose**: Combines multiple metrics with different weights
**Use Case**: When you want to balance multiple factors

#### Multiplicative Scoring
**Formula**: `Score * (Amount spent / 1000) * Proposals`
**Purpose**: Emphasizes jobs that score high on ALL dimensions
**Range**: 0 to very large numbers
**Use Case**: When you want to find jobs that are excellent across all metrics

#### Geometric Mean
**Formula**: `(Score * Amount spent * Proposals) ** (1/3)`
**Purpose**: Balanced combination that penalizes low scores in any dimension
**Range**: 0 to moderate numbers
**Use Case**: When you want balanced performance across all metrics

### 4. Categorical Scoring

#### Binary Classification
**Formula**: `(Score > threshold).astype(int)`
**Purpose**: Converts continuous scores to binary categories
**Range**: {0, 1}
**Use Case**: When you need simple yes/no decisions

#### Multi-Class Classification
**Formula**: 
```python
((Score >= 0) & (Score < 20)) * 1 + 
((Score >= 20) & (Score < 40)) * 2 + 
((Score >= 40) & (Score < 60)) * 3 + 
(Score >= 60) * 4
```
**Purpose**: Creates multiple categories with equal intervals
**Range**: {1, 2, 3, 4}
**Use Case**: When you need multiple quality levels

#### Custom Binning
**Formula**: `pd.cut(Score, bins=[0, 20, 40, 60, 100], labels=['Low', 'Medium', 'High', 'Very High'])`
**Purpose**: Creates custom categories with meaningful labels
**Range**: Categorical labels
**Use Case**: When you need interpretable categories

### 5. Financial Analysis Formulas

#### Return on Investment (ROI)
**Formula**: `(Score * Amount spent) / 100000`
**Purpose**: Measures return relative to investment
**Range**: 0 to very large numbers
**Use Case**: When you want to maximize value per dollar

#### Budget Efficiency
**Formula**: `(Score / Amount spent) * 1000`
**Purpose**: Measures score per $1000 spent
**Range**: 0 to very large numbers
**Use Case**: When you want to find cost-effective opportunities

#### Spend per Proposal
**Formula**: `Amount spent / Proposals`
**Purpose**: Average budget per proposal
**Range**: 0 to very large numbers
**Use Case**: When you want to understand budget distribution

#### Value Score
**Formula**: `Score * (Amount spent / 1000)`
**Purpose**: Combines quality and budget
**Range**: 0 to very large numbers
**Use Case**: When you want to balance quality and budget

### 6. Geographic Scoring

#### Country Mapping
**Formula**: 
```python
Country.map({
    'United States': 100, 
    'UAE': 90, 
    'Canada': 80, 
    'UK': 70
}).fillna(50)
```
**Purpose**: Assigns scores based on country
**Range**: 0-100
**Use Case**: When you have preferences for certain countries

#### Geographic Premium
**Formula**: `(Country == 'United States').astype(int) * 20`
**Purpose**: Adds bonus points for preferred countries
**Range**: 0 or 20
**Use Case**: When you want to boost scores for specific countries

### 7. Text Analysis Formulas

#### String Matching
**Formula**: `Job Title.str.contains('Python', case=False)`
**Purpose**: Identifies jobs containing specific keywords
**Range**: {True, False}
**Use Case**: When you want to filter by technology or skills

#### Pattern Matching
**Formula**: `Job Title.str.contains('Senior|Lead|Principal', case=False)`
**Purpose**: Identifies jobs matching multiple patterns
**Range**: {True, False}
**Use Case**: When you want to find senior-level positions

#### String Length Analysis
**Formula**: `Job Title.str.len()`
**Purpose**: Measures title length
**Range**: 0 to very large numbers
**Use Case**: When you want to analyze title complexity

#### Word Count
**Formula**: `Job Title.str.split().str.len()`
**Purpose**: Counts words in title
**Range**: 0 to very large numbers
**Use Case**: When you want to analyze title detail level

### 8. Performance Metrics

#### Response Rate
**Formula**: `(Interviewing / Proposals) * 100`
**Purpose**: Percentage of proposals that get interviews
**Range**: [0, 100]
**Use Case**: When you want to measure proposal success rate

#### Engagement Score
**Formula**: `Proposals + Interviewing + Invite Sent`
**Purpose**: Total engagement across all activities
**Range**: 0 to very large numbers
**Use Case**: When you want to measure overall activity level

#### Success Rate
**Formula**: `(Interviewing / (Proposals + 1)) * 100`
**Purpose**: Success rate with smoothing to avoid division by zero
**Range**: [0, 100]
**Use Case**: When you want to measure success with small samples

#### Competition Level
**Formula**: `Proposals / (Amount spent / 1000 + 1)`
**Purpose**: Proposals per $1000 budget with smoothing
**Range**: 0 to very large numbers
**Use Case**: When you want to measure competition intensity

### 9. Time-Based Analysis

#### Member Duration
**Formula**: `Member since.apply(lambda x: (datetime.now() - x).days if pd.notna(x) else 0)`
**Purpose**: Days since member joined
**Range**: 0 to very large numbers
**Use Case**: When you want to analyze member experience

#### New Member Detection
**Formula**: `Member since.apply(lambda x: (datetime.now() - x).days <= 30 if pd.notna(x) else False)`
**Purpose**: Identifies new members (last 30 days)
**Range**: {True, False}
**Use Case**: When you want to find new opportunities

#### Join Year
**Formula**: `Member since.dt.year`
**Purpose**: Year when member joined
**Range**: 2000 to current year
**Use Case**: When you want to analyze trends by join year

### 10. Advanced Statistical Formulas

#### Moving Average
**Formula**: `Score.rolling(window=5).mean()`
**Purpose**: Smooths out short-term fluctuations
**Range**: Same as original data
**Use Case**: When you want to identify trends

#### Standard Deviation
**Formula**: `Score.std()`
**Purpose**: Measures data variability
**Range**: 0 to very large numbers
**Use Case**: When you want to understand data spread

#### Coefficient of Variation
**Formula**: `Score.std() / Score.mean()`
**Purpose**: Relative variability measure
**Range**: 0 to very large numbers
**Use Case**: When you want to compare variability across different scales

#### Skewness
**Formula**: `Score.skew()`
**Purpose**: Measures asymmetry of distribution
**Range**: -∞ to +∞
**Use Case**: When you want to understand data distribution shape

### 11. Boolean Logic Formulas

#### AND Operations
**Formula**: `(Score > 30) & (Amount spent > 20000)`
**Purpose**: Both conditions must be true
**Range**: {True, False}
**Use Case**: When you want to find jobs meeting multiple criteria

#### OR Operations
**Formula**: `(Country == 'United States') | (Country == 'UAE')`
**Purpose**: Either condition can be true
**Range**: {True, False}
**Use Case**: When you want to find jobs from multiple countries

#### NOT Operations
**Formula**: `Country != 'Unknown'`
**Purpose**: Condition must be false
**Range**: {True, False}
**Use Case**: When you want to exclude certain values

#### Complex Boolean Logic
**Formula**: `(Score > 30) & (Amount spent > 20000) & (Country != 'Unknown')`
**Purpose**: Multiple conditions with different operators
**Range**: {True, False}
**Use Case**: When you want complex filtering criteria

### 12. Error Handling in Formulas

#### Safe Division
**Formula**: `(Interviewing / (Proposals + 1)) * 100`
**Purpose**: Prevents division by zero
**Range**: [0, 100]
**Use Case**: When you have potential zero denominators

#### NaN Handling
**Formula**: `df['column'].fillna(0)`
**Purpose**: Replaces missing values with default
**Range**: Same as original data
**Use Case**: When you have missing data

#### Type Conversion
**Formula**: `pd.to_numeric(df['column'], errors='coerce')`
**Purpose**: Converts to numeric, errors become NaN
**Range**: Numeric or NaN
**Use Case**: When you have mixed data types

### 13. Performance Optimization

#### Vectorized Operations
**Formula**: `df['new_column'] = df['col1'] + df['col2']`
**Purpose**: Fast element-wise operations
**Range**: Same as input data
**Use Case**: When you need fast calculations

#### Boolean Indexing
**Formula**: `df[df['Score'] > 30]`
**Purpose**: Fast filtering
**Range**: Subset of original data
**Use Case**: When you need to filter data efficiently

#### Chained Operations
**Formula**: `df['col'].str.strip().str.lower()`
**Purpose**: Multiple string operations in sequence
**Range**: Processed string data
**Use Case**: When you need multiple string transformations

### 14. Formula Validation

#### Syntax Validation
**Formula**: `df.eval(expression, engine='python')`
**Purpose**: Validates expression syntax
**Range**: Same as expression result
**Use Case**: When you want to test formulas before applying

#### Type Validation
**Formula**: `pd.api.types.is_numeric_dtype(result)`
**Purpose**: Checks if result is numeric
**Range**: {True, False}
**Use Case**: When you need to validate data types

#### Range Validation
**Formula**: `result.between(0, 100)`
**Purpose**: Checks if values are in expected range
**Range**: {True, False}
**Use Case**: When you need to validate result ranges

### 15. Formula Testing Framework

#### Unit Testing
**Formula**: `test_single_formula(df, formula, name)`
**Purpose**: Tests individual formulas
**Range**: Test results
**Use Case**: When you want to validate formula correctness

#### Integration Testing
**Formula**: `run_experiment(df, name, control_filter, treatment_filter, metric)`
**Purpose**: Tests formula combinations
**Range**: Experiment results
**Use Case**: When you want to test formula interactions

#### Performance Testing
**Formula**: `%timeit df.eval(expression)`
**Purpose**: Measures formula execution time
**Range**: Time measurements
**Use Case**: When you want to optimize formula performance

---

## Formula Best Practices

### 1. Always Validate Inputs
- Check for NaN values
- Verify data types
- Ensure expected ranges

### 2. Use Vectorized Operations
- Avoid loops when possible
- Use pandas methods
- Leverage numpy functions

### 3. Handle Edge Cases
- Division by zero
- Empty datasets
- Missing values

### 4. Document Formulas
- Explain purpose
- Define ranges
- Provide examples

### 5. Test Thoroughly
- Unit tests for individual formulas
- Integration tests for combinations
- Performance tests for optimization

---

*This technical documentation provides the mathematical foundations for all formulas used in the Ultimate Dashboard. Understanding these concepts will help you create more effective custom formulas and troubleshoot issues.*

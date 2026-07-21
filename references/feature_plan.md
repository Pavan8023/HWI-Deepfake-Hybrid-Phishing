# Feature Engineering Plan

## Research project

**Title:** Deepfake-Enabled Hybrid Phishing Detection Through a Human Weakness Index

**Researcher:** Pavan Patil  
**Programme:** MSc Information Systems with Computing  
**Institution:** Dublin Business School

## Purpose

This document defines the features that will be extracted from the four approved dataset tracks:

1. Awareness interactions
2. Human-generated and LLM-generated emails
3. Main phishing-email corpora
4. Malicious URLs

The datasets remain separate because their rows represent different units of analysis.

The feature engineering process must not:

- treat an email as a human participant;
- treat a URL as a human participant;
- merge unrelated datasets row-by-row;
- use identifiers as predictive features;
- create target leakage;
- modify raw data;
- claim that proxy variables are direct psychological measurements.

---

# 1. Awareness dataset

## Source file

`data/raw/awareness/phishing_awareness_dataset.csv`

## Unit of analysis

One recorded email interaction or user-email interaction.

This requires final confirmation from the dataset documentation.

## Candidate outcome

`clicked_link`

Provisional interpretation:

- `yes` or `1`: the link was clicked;
- `no` or `0`: the link was not clicked.

The exact label meaning must be confirmed before modelling.

## Original variables

| Variable | Type | Proposed role | Action |
|---|---|---|---|
| user_id | Identifier | Record/user identifier | Remove from modelling |
| email_subject | Text | Persuasion and contextual features | Retain |
| sender_email_domain | Text/category | Sender context | Transform |
| hover_time_ms | Numeric | Interaction/hesitation proxy | Retain |
| clicked_link | Binary | Candidate outcome | Retain as target |
| reported_email | Binary | Security-response proxy | Confirm timing |
| device_type | Category | Interaction context | Encode |
| browser_used | Category | Interaction context | Encode |
| email_received_time | Date/time | Time-context features | Transform |
| session_duration_sec | Numeric | Interaction-duration proxy | Retain |
| geo_location | Precise location | Sensitive high-cardinality field | Exclude initially |
| email_language | Category | Language context | Encode |

## Planned engineered features

| Feature | Source | Definition | Research role |
|---|---|---|---|
| hover_time_seconds | hover_time_ms | hover_time_ms / 1000 | Behavioural interaction |
| session_duration_minutes | session_duration_sec | session_duration_sec / 60 | Behavioural interaction |
| hover_session_ratio | hover_time_ms, session_duration_sec | hover seconds divided by session seconds | Hesitation/attention proxy |
| reported_binary | reported_email | Convert yes/no to 1/0 | Security-response proxy |
| received_hour | email_received_time | Hour from 0 to 23 | Context |
| received_day_of_week | email_received_time | Monday–Sunday | Context |
| received_weekend | email_received_time | 1 for Saturday/Sunday | Context |
| received_time_period | email_received_time | Morning/afternoon/evening/night | Context |
| subject_character_count | email_subject | Number of characters | Message complexity |
| subject_word_count | email_subject | Number of words | Message complexity |
| subject_url_count | email_subject | Number of URL patterns | Call-to-action proxy |
| subject_exclamation_count | email_subject | Number of `!` symbols | Urgency proxy |
| subject_question_count | email_subject | Number of `?` symbols | Persuasion proxy |
| subject_uppercase_ratio | email_subject | Uppercase letters divided by alphabetic letters | Emphasis proxy |
| subject_urgency_count | email_subject | Count of urgency keywords | Urgency proxy |
| subject_authority_count | email_subject | Count of authority keywords | Authority proxy |
| subject_financial_count | email_subject | Count of financial keywords | Financial-lure proxy |
| subject_fear_count | email_subject | Count of fear/threat keywords | Fear proxy |
| domain_length | sender_email_domain | Character length | Sender-context feature |
| domain_digit_count | sender_email_domain | Number of digits | Suspicious-domain proxy |
| domain_hyphen_count | sender_email_domain | Number of hyphens | Suspicious-domain proxy |
| domain_subdomain_count | sender_email_domain | Number of dots/subdomain separators | Sender complexity |
| domain_free_email_indicator | sender_email_domain | Whether domain belongs to a common free-email provider | Contextual signal |

## Leakage warning

`reported_email` may have occurred after clicking. It must not be used to predict `clicked_link` until the event order is confirmed.

---

# 2. Human and LLM email dataset

## Source files

Human-generated:

- `data/raw/ai_emails/human-generated/legit.csv`
- `data/raw/ai_emails/human-generated/phishing.csv`

LLM-generated:

- `data/raw/ai_emails/llm-generated/legit.csv`
- `data/raw/ai_emails/llm-generated/phishing.csv`

## Units of analysis

One email or generated email message.

## Experimental labels

Two dimensions will be created from the source folders:

### Source label

- `human`
- `llm`

### Email class

- `legitimate`
- `phishing`

The original `label` values must be checked because the AI legitimate file appeared to contain label value `1`. Folder-based class labels may therefore be more reliable than the raw label column.

## Common text construction

For human emails:

`combined_text = subject + body`

For LLM emails:

`combined_text = text`

The same engineered features will be extracted from `combined_text` to create a fair comparison.

## Planned text features

| Feature | Definition | Research role |
|---|---|---|
| character_count | Number of characters | Message length |
| word_count | Number of word tokens | Message length |
| sentence_count | Number of sentences | Structure |
| average_word_length | Mean word length | Linguistic complexity |
| average_sentence_length | Words divided by sentences | Linguistic complexity |
| unique_word_count | Number of unique words | Vocabulary diversity |
| lexical_diversity | Unique words divided by total words | Vocabulary diversity |
| uppercase_character_count | Number of uppercase letters | Emphasis |
| uppercase_ratio | Uppercase letters divided by alphabetic characters | Emphasis |
| digit_count | Number of digits | Financial/account content |
| exclamation_count | Number of exclamation marks | Urgency/emotion |
| question_count | Number of question marks | Engagement/persuasion |
| comma_count | Number of commas | Writing style |
| newline_count | Number of line breaks | Formatting style |
| url_count | Number of URL patterns | Action-request indicator |
| email_address_count | Number of email-address patterns | Contact/impersonation indicator |
| currency_symbol_count | Count of €, $, £ and similar symbols | Financial-lure indicator |
| urgency_keyword_count | Count of urgency words | Urgency score |
| authority_keyword_count | Count of authority/institution words | Authority score |
| fear_keyword_count | Count of fear and threat terms | Fear score |
| financial_keyword_count | Count of payment/refund/account terms | Financial-lure score |
| credential_keyword_count | Count of password/login/verification terms | Credential-request score |
| reward_keyword_count | Count of prize/reward/opportunity terms | Reward score |
| trust_keyword_count | Count of secure/official/verified terms | Trust-building score |
| call_to_action_count | Count of click/open/verify/update/download terms | Action pressure |
| greeting_present | Whether a greeting is present | Message structure |
| signoff_present | Whether a closing/signature is present | Message structure |
| personalisation_indicator | Presence of recipient-style names/titles | Personalisation |
| suspicious_link_indicator | Presence of URL-like text | Technical/context signal |

## Future optional features

These may require extra packages and must be tested for Python 3.13 compatibility:

- Flesch reading ease
- Flesch–Kincaid grade level
- sentiment polarity
- emotion categories
- TF-IDF vectors

They should not replace interpretable handcrafted features.

---

# 3. Main phishing-email corpora

## Files

- CEAS_08.csv
- Enron.csv
- Ling.csv
- Nazario.csv
- Nigerian_Fraud.csv
- phishing_email.csv
- SpamAssasin.csv

## Unit of analysis

One email.

## Main challenge

The corpora have different schemas and potentially different definitions of:

- legitimate;
- spam;
- fraud;
- phishing.

Spam must not automatically be treated as phishing without confirming the source label meaning.

## Planned harmonised fields

| Harmonised field | Possible source columns |
|---|---|
| email_text | body, text, message, content |
| email_subject | subject, title |
| source_corpus | filename/corpus name |
| original_label | original dataset label |
| binary_phishing_label | confirmed harmonised class |
| sender | sender/from |
| receiver | receiver/to |
| date | date/timestamp |
| url_count_original | urls or URL-count field |

## Planned features

Use the same common email features listed in Section 2.

Additional fields:

| Feature | Definition |
|---|---|
| source_corpus | Originating email corpus |
| subject_available | Whether subject text exists |
| sender_available | Whether sender metadata exists |
| body_missing_indicator | Whether body/message text is missing |
| duplicate_text_hash | Hash used to detect duplicate emails |

## Duplicate handling

Duplicate messages must be detected across all corpora before combining any email datasets.

---

# 4. Malicious URL dataset

## Source file

`data/raw/phishing_urls/malicious_phish.csv`

## Unit of analysis

One URL.

## Expected target

A URL class such as:

- benign
- phishing
- malware
- defacement

The exact column and labels must be confirmed.

## Planned URL features

| Feature | Definition | Research role |
|---|---|---|
| url_length | Total number of characters | Complexity |
| hostname_length | Hostname length | Domain complexity |
| path_length | URL path length | Structural complexity |
| query_length | Query-string length | Tracking/parameter complexity |
| fragment_length | Fragment length | Structural feature |
| dot_count | Number of dots | Subdomain complexity |
| hyphen_count | Number of hyphens | Domain/path complexity |
| underscore_count | Number of underscores | Structural feature |
| slash_count | Number of `/` characters | Path complexity |
| question_mark_count | Number of `?` characters | Query indicator |
| equals_count | Number of `=` characters | Parameter indicator |
| ampersand_count | Number of `&` characters | Multi-parameter indicator |
| at_symbol_count | Number of `@` characters | Obfuscation indicator |
| percent_symbol_count | Number of `%` characters | Encoding indicator |
| digit_count | Number of digits | Obfuscation indicator |
| digit_ratio | Digits divided by URL length | Obfuscation indicator |
| special_character_count | Non-alphanumeric characters | Complexity |
| https_indicator | Whether scheme is HTTPS | Technical-context feature |
| http_indicator | Whether scheme is HTTP | Technical-context feature |
| ip_address_indicator | Whether hostname is an IP address | Suspiciousness |
| punycode_indicator | Presence of `xn--` | Homograph-risk indicator |
| shortened_url_indicator | Known URL-shortener domain | Redirection risk |
| suspicious_keyword_count | login, verify, secure, account, update, etc. | Phishing-content indicator |
| subdomain_count | Number of hostname components | Domain complexity |
| top_level_domain | Extracted TLD where available | Domain feature |
| url_entropy | Character-level entropy | Randomness/obfuscation |
| repeated_character_indicator | Long repeated character sequences | Obfuscation |
| encoded_character_indicator | Presence of percent-encoded sequences | Obfuscation |

## Output naming

This dataset produces:

- Technical Threat Score
- Technical Exposure Risk

It does not independently produce a Human Weakness Index.

---

# 5. HWI relationship

The HWI will not be calculated in the feature-engineering phase.

Potential dimensions supported by current datasets are:

- Behavioural Interaction Risk
- Security Response Risk
- Contextual Exposure Risk
- Persuasion Risk
- Trust-Manipulation Risk
- Urgency/Fear Risk

The exact HWI formula and weights will be designed only after:

1. feature extraction;
2. feature distribution analysis;
3. target validation;
4. leakage checks;
5. literature justification.
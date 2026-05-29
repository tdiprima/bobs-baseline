# 🍔 Bob's Baseline

**AKA "Bob's Burgers ML Project"**

Learning what "normal orders" look like — and flagging the ones that make Bob sigh.

## What this is

This project explores **order / request anomaly detection** using unsupervised machine learning.

The idea is simple: instead of labeling things as *good* or *bad*, the system learns what **normal operational behavior** looks like and highlights orders that are unusually large, fast, expensive, or disruptive.

Inspired by *Bob's Burgers*, built with blue-team instincts.

## The Bob's Burgers mental model

At Bob's Burgers, most orders are predictable:

* Teddy orders the same thing every day
* Lunch rush looks like lunch rush
* Prep times stay within reason

Then something happens:

* A massive order right before closing
* A customer ordering way faster than usual
* An order that explodes prep time and cost

Bob doesn't ask:

"Is this malicious?"

Bob asks:

"Is this order gonna be a problem?"

That's the exact question this project answers.

## Why this exists

In real systems, many incidents start as *operational anomalies*:

* Sudden cost spikes
* Abusive API usage
* Runaway jobs
* Resource exhaustion

Rule-based alerts are brittle. This project focuses on **pattern-based detection** instead.

## What the model looks at

Example features (restaurant-flavored, but transferable):

* items\_per\_order
* time\_since\_last\_order
* order\_hour
* customer\_order\_frequency
* estimated\_prep\_time
* total\_order\_cost

No identities required. No payload inspection. Just behavior.

## How it works

1. Ingest order or request events
2. Learn baseline behavior
3. Score new events by how unusual they are
4. Surface the ones likely to cause trouble

Models explored:

* DBSCAN


## How this maps to real systems

Swap "orders" for:

* API requests
* Background jobs
* Cloud workloads
* Billing events

Same logic. Same detection strategy.

## Example question this answers

"Would this request make the on-call engineer sigh?"

If yes, it probably deserves a closer look.

## What this is *not*

* ❌ A fraud engine
* ❌ A production billing system
* ❌ A deep learning project

This is a **learning-focused anomaly detection lab**.

## Disclaimer

All data is synthetic or anonymized. No real customers, no real transactions.

<br>

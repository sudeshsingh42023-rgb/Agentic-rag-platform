# ENTERPRISE DATA RETENTION AND CLASSIFICATION STANDARD
Standard reference ERS-014. Owner: Office of the Chief Information Security Officer.

## 1. PURPOSE AND SCOPE
1.1 This standard defines minimum retention periods and handling controls for data held by the organisation and its processors.
1.2 It applies to all business units, contractors and third-party processors.

## 2. CLASSIFICATION TIERS
2.1 RESTRICTED covers data whose disclosure would cause severe harm, including payment card data, biometric identifiers and government identification numbers.
2.2 CONFIDENTIAL covers customer records, contracts, pricing and unreleased financial results.
2.3 INTERNAL covers operational documentation not intended for publication.
2.4 PUBLIC covers material approved for external release.

## 3. RETENTION PERIODS
3.1 Financial transaction records shall be retained for 8 years from the end of the relevant financial year.
3.2 Employee personnel files shall be retained for 7 years after the termination of employment.
3.3 Customer support transcripts shall be retained for 24 months.
3.4 CCTV footage shall be retained for 90 days unless preserved under a legal hold.
3.5 System and application audit logs shall be retained for 18 months, with the most recent 90 days held in hot storage.

## 4. ENCRYPTION AND ACCESS
4.1 RESTRICTED data shall be encrypted at rest using AES-256 and in transit using TLS 1.2 or higher.
4.2 Access to RESTRICTED data requires a documented business justification and approval from the data owner, reviewed every 90 days.
4.3 Privileged access sessions shall be recorded and reviewed on a monthly basis.

## 5. DELETION AND LEGAL HOLD
5.1 Data that has reached the end of its retention period shall be deleted within 30 days.
5.2 Deletion of RESTRICTED data requires cryptographic erasure and a certificate of destruction.
5.3 A legal hold suspends all deletion obligations for the affected records until the hold is formally released by the legal department.

## 6. INCIDENT REPORTING
6.1 Any suspected loss of RESTRICTED data shall be reported to the security operations centre within 1 hour of discovery.
6.2 A preliminary incident report shall be filed within 24 hours and a root cause analysis within 10 working days.
6.3 Regulatory notification obligations shall be assessed within 72 hours of confirming a personal data breach.

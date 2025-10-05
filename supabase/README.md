# Supabase SSL Certificate

This folder contains the SSL certificate for secure PostgreSQL connections to Supabase.

## Certificate File

Place your Supabase SSL certificate in this folder with the name `prod-ca-2021.crt`.

You can download the certificate from your Supabase project dashboard:
1. Go to Settings > Database
2. Scroll down to "Connection parameters"
3. Download the SSL certificate

## Configuration

The certificate path can be configured using the environment variable:
```bash
POSTGRES_SSL_CERT_PATH=supabase/prod-ca-2021.crt
```

## SSL Mode

You can also configure the SSL mode:
```bash
POSTGRES_SSL_MODE=require  # Default: require
```

Available SSL modes:
- `disable` - No SSL
- `allow` - Try SSL, fallback to non-SSL
- `prefer` - Try SSL first, fallback to non-SSL
- `require` - Require SSL (default)
- `verify-ca` - Require SSL and verify certificate
- `verify-full` - Require SSL, verify certificate and hostname

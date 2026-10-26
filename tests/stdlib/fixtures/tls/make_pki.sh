#!/usr/bin/env bash
# Regenerates the TLS test PKI in this directory (needs openssl 1.1+):
#   root.pem         RSA-2048 root CA (self-signed, 100 years)
#   int.pem          P-384 intermediate CA, signed by root (sha256WithRSA)
#   leaf-ec.pem/.key P-256 server cert (localhost, *.test.example, 127.0.0.1, ::1),
#                    signed by int (ecdsa-with-SHA384); chain-ec.pem = leaf + int
#   leaf-rsa.pem/.key RSA-2048 server cert (localhost, 127.0.0.1), signed by
#                    root with RSASSA-PSS/SHA-256
#   expired.pem/.key P-256 localhost cert signed by int, valid 2020 only
#   other-root.pem   an unrelated root; other-leaf.pem/.key a localhost cert from it
#   self.pem/.key    a self-signed localhost cert
#   client.pem/.key  P-256 client cert (clientAuth), signed by int
# The tests use these files as checked in; rerun only to change them.
set -euo pipefail
cd "$(dirname "$0")"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
O=openssl
cat > "$D/ca.ext" <<X
basicConstraints=critical,CA:TRUE
keyUsage=critical,keyCertSign,cRLSign
subjectKeyIdentifier=hash
X
cat > "$D/leaf.ext" <<X
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:localhost,DNS:*.test.example,IP:127.0.0.1,IP:::1
X
cat > "$D/leafrsa.ext" <<X
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:localhost,IP:127.0.0.1
X
cat > "$D/client.ext" <<X
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=clientAuth
X
$O req -x509 -newkey rsa:2048 -nodes -keyout root.key -out root.pem -days 36500 -sha256 \
    -subj "/C=US/O=SafeC Test/CN=SafeC Test Root" -addext basicConstraints=critical,CA:TRUE \
    -addext keyUsage=critical,keyCertSign,cRLSign 2>/dev/null
$O ecparam -name secp384r1 -genkey -noout -out int.key
$O req -new -key int.key -subj "/C=US/O=SafeC Test/CN=SafeC Test Intermediate" -out "$D/int.csr"
$O x509 -req -in "$D/int.csr" -CA root.pem -CAkey root.key -CAcreateserial -days 36000 -sha256 \
    -extfile "$D/ca.ext" -out int.pem 2>/dev/null
$O ecparam -name prime256v1 -genkey -noout -out "$D/leaf-ec.sec1"
$O pkcs8 -topk8 -nocrypt -in "$D/leaf-ec.sec1" -out leaf-ec.key
$O req -new -key leaf-ec.key -subj "/CN=localhost" -out "$D/leaf.csr"
$O x509 -req -in "$D/leaf.csr" -CA int.pem -CAkey int.key -CAcreateserial -days 35000 -sha384 \
    -extfile "$D/leaf.ext" -out leaf-ec.pem 2>/dev/null
cat leaf-ec.pem int.pem > chain-ec.pem
$O genrsa -out leaf-rsa.key 2048 2>/dev/null
$O req -new -key leaf-rsa.key -subj "/CN=localhost" -out "$D/leafrsa.csr"
$O x509 -req -in "$D/leafrsa.csr" -CA root.pem -CAkey root.key -CAcreateserial -days 35000 -sha256 \
    -sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:32 -extfile "$D/leafrsa.ext" -out leaf-rsa.pem 2>/dev/null
# expired: openssl ca can set explicit dates
mkdir -p "$D/ca"; touch "$D/ca/index.txt"; echo 1000 > "$D/ca/serial"
cat > "$D/ca.cnf" <<X
[ca]
default_ca = d
[d]
dir = $D/ca
database = \$dir/index.txt
serial = \$dir/serial
new_certs_dir = \$dir
default_md = sha256
policy = p
copy_extensions = none
unique_subject = no
[p]
commonName = supplied
X
$O ecparam -name prime256v1 -genkey -noout -out expired.key
$O req -new -key expired.key -subj "/CN=localhost" -out "$D/exp.csr"
$O ca -batch -config "$D/ca.cnf" -cert int.pem -keyfile int.key -in "$D/exp.csr" -out "$D/exp.pem" \
    -startdate 20200101000000Z -enddate 20201231000000Z -extfile "$D/leaf.ext" -notext 2>/dev/null
$O x509 -in "$D/exp.pem" -out expired.pem
cat expired.pem int.pem > chain-expired.pem
$O req -x509 -newkey rsa:2048 -nodes -keyout "$D/other-root.key" -out other-root.pem -days 36500 -sha256 \
    -subj "/CN=Other Root" -addext basicConstraints=critical,CA:TRUE -addext keyUsage=critical,keyCertSign 2>/dev/null
$O ecparam -name prime256v1 -genkey -noout -out other-leaf.key
$O req -new -key other-leaf.key -subj "/CN=localhost" -out "$D/ol.csr"
$O x509 -req -in "$D/ol.csr" -CA other-root.pem -CAkey "$D/other-root.key" -CAcreateserial -days 35000 -sha256 \
    -extfile "$D/leaf.ext" -out other-leaf.pem 2>/dev/null
$O req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -keyout self.key -out self.pem -days 35000 \
    -subj "/CN=localhost" -addext subjectAltName=DNS:localhost,IP:127.0.0.1 2>/dev/null
$O ecparam -name prime256v1 -genkey -noout -out client.key
$O req -new -key client.key -subj "/CN=SafeC Test Client" -out "$D/cl.csr"
$O x509 -req -in "$D/cl.csr" -CA int.pem -CAkey int.key -CAcreateserial -days 35000 -sha384 \
    -extfile "$D/client.ext" -out client.pem 2>/dev/null
cat client.pem int.pem > chain-client.pem
rm -f *.srl root.key int.key
echo "PKI written to $(pwd)"

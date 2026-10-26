// extracted from safec-docs reference/concurrency.md (block 2)
?&int worker(?&int arg) { return null; }

int main() {
    long long h = spawn(worker, null);  // OK: worker is static
    join(h);
    return 0;
}

# cremi-deform
Introduce distortions to labeled volumes

## Build instructions:

Clone the repositories for `bigcat` (branch `cremi-inspect`) and `imglib2`:

* https://github.com/saalfeldlab/bigcat.git
* https://github.com/imglib/imglib2.git

In each repository, run `mvn install`. Afterwards, run

```
mvn clean compile assembly:single
```

in this directory to get a single `.jar` file in `targets`.

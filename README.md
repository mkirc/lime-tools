# lime wip

## setup devcontainer

```bash
cd devcontainer/
docker compose build
```
builds an image 'lime-dev' for dependency isolation

## running lime

`docker run -it -v ./lime-1.9.5:/lime -v [PATH TO MODEL]:/model lime-dev lime-run`


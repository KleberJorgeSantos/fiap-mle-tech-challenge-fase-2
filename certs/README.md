# Certificados adicionais para o build da imagem

Redes com **inspeção TLS** (proxy corporativo, antivírus como Avast/Kaspersky,
firewall de borda) reassinam o tráfego HTTPS com uma autoridade própria. Dentro
do container essa autoridade é desconhecida, e o `pip`/`poetry` falha com:

```
SSLError: certificate verify failed: unable to get local issuer certificate
```

**Solução:** coloque o certificado raiz da sua rede aqui, com extensão `.crt`
(formato PEM). O `Dockerfile` instala tudo que estiver neste diretório como
raiz confiável antes de baixar qualquer dependência.

```bash
# Exemplo — Avast no Windows:
cp "/c/ProgramData/Avast Software/Avast/wscert.pem" certs/avast.crt
```

Os arquivos `.crt` são ignorados pelo git (são específicos de cada máquina).
**Com o diretório vazio o build funciona normalmente** — em uma rede sem
inspeção, nada aqui é necessário.

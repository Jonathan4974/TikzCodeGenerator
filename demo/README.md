# Demo

End-to-end demo of the final model — takes a sketch as input 
and outputs compiled TikZ code and a rendered figure.

## Enroot: Ollama Server

```bash
cd tikzcodegenerator/demo

enroot import \
  --output ollama-server.sqsh \
  docker://ollama/ollama:latest
```

## Run demo on the cluster

change the model name and user name, then launch the demo on the cluster

```bash
export MODEL=gemma4:31b-it-q4_K_M
export I9_USER=s0042

sbatch run_demo.sbatch
```

## Open frontend in the local browser
- open logs/demo-%j.out
- following the instruction to build the connection and open the demo in your browser:
    ```text
    ======================================================
    ✅ Demo API is starting!
    🔗 Access URL inside cluster: http://${NODE_IP}:${PORT}
    ======================================================
    To access from your local browser, run this SSH tunnel:
        ssh -N -L ${PORT_LOCAL}:${NODE_IP}:${PORT} ${USER}@${LOGIN_NODE}
    Then open http://localhost:${PORT_LOCAL} in your browser.
    ======================================================
    ```

## Demo debugging
you can also test the demo with a small model on the workstation before submit
- run demo on the workstation
     ```bash
    export MODEL=gemma4:31b-it-q4_K_M
    export I9_USER=s0042

    bash ./run_demo_debug.sh
    ```
- following the instruction to build the connection and open the demo in your browser:
    ```text
    ======================================================
    ✅ Demo API is starting!
    🔗 Access URL inside workstation: http://${NODE_IP}:${PORT}
    ======================================================
    To access from your local browser, run this SSH tunnel:
       ssh -N -L ${PORT_LOCAL}:${NODE_IP}:${PORT} ${USER}@$(hostname -s)
    Then open http://localhost:${PORT_LOCAL} in your browser.
    ======================================================
    ```
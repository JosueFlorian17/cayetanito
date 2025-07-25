def construir_prompt(pregunta, config_aula, contexto_externo=None):
    prompt = "<s>[INST]\n"
    prompt += f"Aula: {config_aula.get('aula', 'No especificado')}\n"
    prompt += f"Palabra clave de asamblea: {config_aula.get('palabra_clave_asamblea', 'Ninguna')}\n"

    temas = ", ".join(config_aula.get("temas_aprendidos", []))
    prompt += f"Temas aprendidos: {temas or 'Ninguno'}\n"

    preferencias = ", ".join(config_aula.get("preferencias_locales", []))
    prompt += f"Preferencias locales: {preferencias or 'Ninguna'}\n"

    if contexto_externo:
        prompt += f"Información adicional: {contexto_externo.strip()}\n"

    prompt += f"\nPregunta: {pregunta.strip()}\n"
    prompt += "[/INST]"
    return prompt

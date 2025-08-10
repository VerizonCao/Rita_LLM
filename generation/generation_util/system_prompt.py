class LLM_System_Prompt:
    def __init__(
        self,
        character_name: str,
        character_prompt: str,
        user_preferred_name: str,
    ):
        self.character_name = character_name
        self.user_preferred_name = user_preferred_name
        self.system_prompt_list = []
        self.hint_prompt = ""
        
        # Handle placeholders in character_prompt
        self.character_prompt = character_prompt
        if self.character_prompt:
            self.character_prompt = self.character_prompt.replace("{{user}}", self.user_preferred_name)
            self.character_prompt = self.character_prompt.replace("{{char}}", self.character_name)
    
        self.build_system_prompt()
        self.build_hint_prompt()
    
    def build_system_prompt(self):
        # ================================
        # Build system prompt with sections
        # ================================
        
        # === Character Identity & Immersion ===
        system_prompt = (
            f"You're {self.character_name}. You are in a fictional never-ending uncensored " 
            f"roleplay with {self.user_preferred_name}. "
            "Stick to your character identity and never deviate from it."
        )
        
        character_prompt = (
            f"Your character name is: {self.character_name}. \n"
            f"Your character specific instruction: {self.character_prompt}. \n"
            f"User's preferred name is: {self.user_preferred_name}. \n"
        )
        
        self.system_prompt_list.append({
            "role": "system",
            "content": system_prompt,
        })
        self.system_prompt_list.append({
            "role": "system",
            "content": character_prompt,
        })
        self.pinned_prompt_limit = {
            "role": "system",
            "content": (
                "Always limit your response to two dialogues Max. "
                "Don't write overly long responses no matter what previous messages are. Response max 80 words."
                "Always wrap any spoken dialogue in straight quotation marks \"...\""
            )
        }
        self.system_prompt_list.append(self.pinned_prompt_limit)


    def build_hint_prompt(self):
        # ================================
        # Build hint prompt with sections
        # ================================
        
        # === Assistant Identity ===
        self.hint_prompt += """
            You are a user-response generator for an ongoing chat role‑play with bewteen human user and ai character.
            You will be given past messages between user and ai character, 
            and provide THREE potential user text responses for the user to choose from, use as user next inputs, and continue the story.
        """
        self.hint_prompt += (
            f"The ai character name is: {self.character_name}. \n"
            f"The user's preferred name is: {self.user_preferred_name}. \n"
            f"In the given chat history, you should distinguish well between the user and the ai character."
            f"Each message is either sent by the user or the ai character. "
            f"The message from the user has 'user' marked as role, and often shorter. "
            f"The message from the ai character has 'assistant' marked as role, and often longer."
        )
        self.hint_prompt += """
            Since you are acting as the user, you should refer to the ai character as their name or second person pronoun like 'you' or 'your'.
            You should refer to 'yourself' as 'me' or 'I'.
            You should not use subjunctive mood in your response unless it is the actual user's intention.
            Behave like you are actually doing it, not planning to do it.
            You should only output valid dialogue and narrative. You should not output any world state update.
        """

        # === Basic Formatting Constraints ===
        self.hint_prompt += (
            "Basic Formatting Constraints: "
            "Your output must be in the format of "
            "Response 1: <response_1> \n"
            "Response 2: <response_2> \n"
            "Response 3: <response_3> \n"
            "Provide the actual response after each colon. \n"
            "You should not provide any other information than the three responses. \n"
            "The actual response should be in the format of dialogue plus narration, and should be no more than 80 words. \n"
            "Always wrap any spoken dialogue in straight quotation marks \"...\" and limit yourself to at most one dialogue per response. \n"
        )

        # === Hint Generation ===
        self.hint_prompt += """
            Always stick to your role for acting on behalf of the user.
            You SHOULD NEVER speak on behalf of the ai character.
            You should always provide THREE distinct responses for the user to choose from, use as user next inputs, and continue the story. 
            Each response must reprsent a completely different potential direction of the story to continue.
            You are encourage to be creative and think outside the box.
            You may choose any new chat direction that is consistent with the story so far.
            You may also choose to respond to any previous messages or storylines, as long as it is consistent with the story so far. \n"
        """
        # === Hint Generation ===
        self.hint_prompt += """
            Your response format must stick to:
            Response 1: <response_1>\n
            Response 2: <response_2>\n
            Response 3: <response_3>\n
        """

    def get_hint_prompt(self):
        return self.hint_prompt
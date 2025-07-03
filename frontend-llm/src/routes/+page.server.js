import { fail } from '@sveltejs/kit';
import Groq from 'groq-sdk';
import { GROQ_API_KEY } from '$env/static/private';

// Initialize the Groq client
const groq = new Groq({
	apiKey: GROQ_API_KEY
});

export const actions = {
	default: async ({ request }) => {
		const formData = await request.formData();
		const message = formData.get('message');
		const model = formData.get('model');

		if (!message || typeof message !== 'string') {
			return fail(400, { error: 'Message cannot be empty.' });
		}
		if (!model || typeof model !== 'string') {
			return fail(400, { error: 'Model not selected.' });
		}

		try {
			const chatCompletion = await groq.chat.completions.create({
				messages: [
					{
						role: 'user',
						content: message
					}
				],
				model: model
			});

			const aiResponse = chatCompletion.choices[0]?.message?.content;

			if (!aiResponse) {
				return fail(500, { error: 'API returned an empty response.' });
			}
			return {
				role: 'assistant',
				content: aiResponse
			};
		} catch (error) {
			console.error('Groq API Error:', error);
			return fail(500, { error: 'Could not connect to the Groq API.' });
		}
	}
};

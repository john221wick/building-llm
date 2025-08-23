<script>
	import { enhance } from '$app/forms';
	import { fly } from 'svelte/transition';
	let showMobileMenu = $state(false);
	let messages = $state([
		{
			role: 'assistant',
			content: "Hello! I'm your assistant. How can I help you today?"
		}
	]);
	let loading = $state(false);
	let inputText = $state('');
	let chatContainer;

	// Model selection state
	const models = [
		{ id: 'llama-3.1-8b-instant', name: 'Llama 3.1 8B Instant 128k' },
		{ id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B Versatile 128k' },
		{
			id: 'meta-llama/llama-4-maverick-17b-128e-instruct',
			name: 'Llama 4 Maverick (17Bx128E) 128k'
		},
		{ id: 'meta-llama/llama-4-scout-17b-16e-instruct', name: 'Llama 4 Scout (17Bx16E) 128k' },
		{ id: 'qwen-qwq-32b', name: 'Qwen 32B 128k' },
		{ id: 'deepseek-r1-distill-llama-70b', name: 'DeepSeek R1 Distill Llama 70B 128k' },
		{ id: 'gemma2-9b-it', name: 'Gemma 2 9B 8k' },
		{ id: 'wick-model', name: 'My Own Custom Model' }
	];
	let currentModel = $state('llama-3.1-8b-instant');

	$effect(() => {
		chatContainer?.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
	});
</script>

<svelte:head>
	<title>Building LLM</title>
</svelte:head>

<div class="flex h-screen flex-col bg-white font-sans dark:bg-[#111111]">
	<!-- replace the whole <header> block with this -->

	<header
		class="flex flex-shrink-0 items-center justify-between border-b border-gray-200 px-4 py-4 sm:px-6 dark:border-gray-800"
	>
		<!-- title -->
		<h1 class="text-lg font-semibold text-gray-900 dark:text-gray-100">Chat</h1>

		<!-- controls (always one row on every screen) -->
		<div class="flex items-center gap-2 sm:gap-2">
			<!-- model selector -->
			<div class="relative">
				<select
					bind:value={currentModel}
					class="appearance-none rounded-md border border-gray-300 bg-white py-1.5 pr-7 pl-2 text-xs text-gray-800 focus:ring-1 focus:ring-gray-400 focus:outline-none sm:pr-8 sm:pl-3 sm:text-sm dark:border-gray-700 dark:bg-[#111111] dark:text-gray-200 dark:focus:ring-gray-600"
				>
					{#each models as model (model.id)}
						<option value={model.id}>{model.name}</option>
					{/each}
				</select>
				<div
					class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-1.5 text-gray-500 sm:px-2"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="12"
						height="12"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"><path d="m6 9 6 6 6-6" /></svg
					>
				</div>
			</div>

			<!-- GitHub icon -->
			<a
				href="https://github.com/john221wick/building-llm"
				target="_blank"
				class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="20"
					height="20"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
				>
					<path
						d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"
					/>
				</svg>
			</a>

			<!-- Docs icon -->
			<a
				href="https://github.com/john221wick/building-llm"
				target="_blank"
				class="text-gray-500 transition-colors hover:text-gray-700 dark:text-white dark:hover:text-gray-300"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="20"
					height="20"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
				>
					<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
					<polyline points="14 2 14 8 20 8" />
					<line x1="16" y1="13" x2="8" y2="13" />
					<line x1="16" y1="17" x2="8" y2="17" />
					<line x1="10" y1="9" x2="8" y2="9" />
				</svg>
			</a>
		</div>
	</header>
	<main bind:this={chatContainer} class="flex-1 overflow-y-auto p-4 pb-28 sm:px-6">
		<div class="mx-auto max-w-3xl space-y-6">
			{#each messages as message, i (message.content + i)}
				<div class="flex items-start gap-4" in:fly={{ y: 20, duration: 400 }}>
					<div
						class="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full border border-gray-200 bg-gray-100 text-gray-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300"
					>
						{#if message.role === 'user'}
							<svg
								xmlns="http://www.w3.org/2000/svg"
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle
									cx="12"
									cy="7"
									r="4"
								/></svg
							>
						{:else}
							<svg
								xmlns="http://www.w3.org/2000/svg"
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><path d="M12 8V4H8" /><rect width="16" height="12" x="4" y="8" rx="2" /><path
									d="M2 14h2"
								/><path d="M20 14h2" /><path d="M15 13v2" /><path d="M9 13v2" /></svg
							>
						{/if}
					</div>
					<div class="flex-1 pt-0.5">
						<div class="mb-2 text-sm font-semibold text-gray-900 dark:text-white">
							{message.role === 'user' ? 'You' : 'Assistant'}
						</div>
						<div class="prose prose-sm max-w-none text-gray-800 dark:text-gray-200">
							{message.content}
						</div>
					</div>
				</div>
			{/each}

			{#if loading}
				<div class="flex animate-pulse items-start gap-4">
					<div
						class="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full border border-gray-200 bg-gray-100 text-gray-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							width="14"
							height="14"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><path d="M12 8V4H8" /><rect width="16" height="12" x="4" y="8" rx="2" /><path
								d="M2 14h2"
							/><path d="M20 14h2" /><path d="M15 13v2" /><path d="M9 13v2" /></svg
						>
					</div>
					<div class="flex-1 pt-1">
						<div class="flex h-8 items-center">
							<div class="flex items-center space-x-1.5 text-gray-500 dark:text-gray-400">
								<div class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400"></div>
								<div
									class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400"
									style="animation-delay: 0.2s"
								></div>
								<div
									class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400"
									style="animation-delay: 0.4s"
								></div>
							</div>
						</div>
					</div>
				</div>
			{/if}
		</div>
	</main>

	<footer class="fixed right-0 bottom-4 left-0 px-4 sm:px-6">
		<div class="mx-auto max-w-3xl">
			<form
				method="POST"
				use:enhance={() => {
					if (!inputText.trim()) return;

					loading = true;
					messages.push({ role: 'user', content: inputText });
					inputText = '';

					return async ({ result }) => {
						if (result.type === 'success' && result.data) {
							messages.push(result.data);
						} else {
							messages.push({ role: 'assistant', content: 'Sorry, an error occurred.' });
						}
						loading = false;
					};
				}}
			>
				<input type="hidden" name="model" value={currentModel} />

				<div
					class="flex items-center rounded-lg border border-gray-300 bg-white p-2 shadow-lg focus-within:ring-1 focus-within:ring-gray-400 dark:border-gray-700 dark:bg-black dark:focus-within:ring-gray-600"
				>
					<textarea
						name="message"
						bind:value={inputText}
						onkeydown={(e) => {
							if (e.key === 'Enter' && !e.shiftKey) {
								e.preventDefault();
								e.currentTarget.form?.requestSubmit();
							}
						}}
						placeholder="Message..."
						class="min-h-[44px] w-full flex-1 resize-none border-0 bg-transparent pl-2 text-gray-900 placeholder-gray-500 focus:ring-0 dark:text-white dark:placeholder-gray-400"
						disabled={loading}
						rows="1"
					></textarea>
					<button
						type="submit"
						class="ml-2 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md text-gray-500 transition-colors hover:bg-gray-100 disabled:cursor-not-allowed disabled:text-gray-300 dark:text-gray-400 dark:hover:bg-gray-800 dark:disabled:text-gray-600"
						disabled={loading || !inputText.trim()}
						aria-label="Send message"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							width="16"
							height="16"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"><path d="m5 12 7-7 7 7" /><path d="M12 19V5" /></svg
						>
					</button>
				</div>
			</form>
		</div>
	</footer>
</div>

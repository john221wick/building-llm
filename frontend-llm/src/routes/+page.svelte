<!-- src/routes/+page.svelte -->
<script>
	let { messages, inputText, loading, models, currentModel } = $state({
		messages: [
			{
				role: 'assistant',
				content: "Hello! I'm your custom LLM assistant. How can I help you today?"
			}
		],
		inputText: '',
		loading: false,
		models: [
			{ id: 'llama3-8b', name: 'Llama 3 8B', icon: 'fa-dragon' },
			{ id: 'mixtral-8x7b', name: 'Mixtral 8x7B', icon: 'fa-brain' },
			{ id: 'custom-model', name: 'Custom Model', icon: 'fa-cogs' }
		],
		currentModel: 'llama3-8b'
	});

	// Simulate API call
	async function sendMessage() {
		if (!inputText.trim() || loading) return;

		const userMessage = { role: 'user', content: inputText };
		messages = [...messages, userMessage];
		inputText = '';
		loading = true;

		// Simulate API delay
		await new Promise((resolve) => setTimeout(resolve, 1500));

		// Simulated response
		const responses = [
			"I understand your question. Based on my knowledge, here's what I can tell you...",
			"That's an interesting question! The answer depends on several factors...",
			"I've analyzed your query and found several relevant points. First...",
			"Great question! Here's a detailed explanation...",
			'I can certainly help with that. Let me break it down for you...'
		];

		const randomResponse = responses[Math.floor(Math.random() * responses.length)];

		messages = [...messages, { role: 'assistant', content: randomResponse }];
		loading = false;
	}

	function handleKeydown(e) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	}
</script>

<div
	class="flex h-screen flex-col bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800"
>
	<!-- Header with model switcher -->
	<header
		class="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-3 dark:border-gray-700 dark:bg-gray-800"
	>
		<div class="flex items-center space-x-2">
			<div
				class="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-r from-blue-600 to-indigo-700"
			>
				<i class="fa-solid fa-brain text-lg text-white"></i>
			</div>
			<h1
				class="bg-gradient-to-r from-blue-600 to-indigo-700 bg-clip-text text-xl font-bold text-transparent"
			>
				Custom LLM
			</h1>
		</div>

		<!-- Model switcher -->
		<div class="flex items-center space-x-2">
			<span class="hidden text-sm text-gray-600 md:block dark:text-gray-400">Model:</span>
			<div class="relative">
				<select
					bind:value={currentModel}
					class="appearance-none rounded-lg border border-gray-300 bg-white py-2 pr-8 pl-3 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none dark:border-gray-600 dark:bg-gray-800"
				>
					{#each models as model (model.id)}
						<option value={model.id}>{model.name}</option>
					{/each}
				</select>
				<div
					class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700 dark:text-gray-300"
				>
					<i class="fa-solid fa-chevron-down text-xs"></i>
				</div>
			</div>
		</div>
	</header>

	<!-- Chat container -->
	<main class="flex-1 overflow-y-auto p-4 pb-24">
		{#if messages.length === 1}
			<div
				class="mx-auto flex h-full max-w-2xl flex-col items-center justify-center px-4 text-center"
			>
				<div class="mb-6 rounded-2xl bg-gradient-to-r from-blue-500 to-indigo-600 p-4">
					<i class="fa-solid fa-robot text-4xl text-white"></i>
				</div>
				<h2 class="mb-2 text-2xl font-bold text-gray-800 dark:text-white">
					Your Custom LLM Interface
				</h2>
				<p class="mb-8 text-gray-600 dark:text-gray-400">
					Start a conversation by typing a message below. Switch between models using the dropdown
					above.
				</p>

				<div class="grid w-full grid-cols-1 gap-4 md:grid-cols-2">
					<div
						class="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800"
					>
						<h3 class="mb-2 font-semibold text-gray-800 dark:text-white">Example Queries</h3>
						<ul class="space-y-2 text-left">
							<li class="text-sm text-gray-600 dark:text-gray-400">
								<i class="fa-solid fa-arrow-right mr-2 text-blue-500"></i>Explain quantum computing
								in simple terms
							</li>
							<li class="text-sm text-gray-600 dark:text-gray-400">
								<i class="fa-solid fa-arrow-right mr-2 text-blue-500"></i>How do I implement a REST
								API in Svelte?
							</li>
							<li class="text-sm text-gray-600 dark:text-gray-400">
								<i class="fa-solid fa-arrow-right mr-2 text-blue-500"></i>What's the weather
								forecast for tomorrow?
							</li>
						</ul>
					</div>

					<div
						class="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800"
					>
						<h3 class="mb-2 font-semibold text-gray-800 dark:text-white">Current Model</h3>
						<div class="flex items-center space-x-3">
							{#each models as model (model.id)}
								{#if model.id === currentModel}
									<div
										class="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-r from-blue-600 to-indigo-700"
									>
										<i class={`fa-solid ${model.icon} text-white`}></i>
									</div>
									<div>
										<div class="font-medium text-gray-900 dark:text-white">{model.name}</div>
										<div class="text-xs text-gray-500 dark:text-gray-400">Ready to chat</div>
									</div>
								{/if}
							{/each}
						</div>
					</div>
				</div>
			</div>
		{:else}
			<div class="mx-auto max-w-3xl space-y-6">
				{#each messages as message, i (i)}
					<div class="flex gap-4 {message.role === 'user' ? 'flex-row-reverse' : ''}">
						{#if message.role === 'assistant'}
							<div
								class="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 text-white"
							>
								<i class="fa-solid fa-robot"></i>
							</div>
						{:else}
							<div
								class="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-gray-800 text-white dark:bg-gray-600"
							>
								<i class="fa-solid fa-user"></i>
							</div>
						{/if}

						<div class="flex-1">
							<div class="mb-1 font-medium text-gray-900 dark:text-white">
								{message.role === 'user' ? 'You' : 'Assistant'}
							</div>
							<div
								class="rounded-xl border border-gray-200 bg-white p-4 text-gray-800 dark:border-gray-700 dark:bg-gray-800/50 dark:text-gray-200"
							>
								{message.content}
							</div>
						</div>
					</div>
				{/each}

				{#if loading}
					<div class="flex gap-4">
						<div
							class="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 text-white"
						>
							<i class="fa-solid fa-robot"></i>
						</div>
						<div class="flex-1">
							<div class="mb-1 font-medium text-gray-900 dark:text-white">Assistant</div>
							<div
								class="flex h-20 items-center justify-center rounded-xl border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800/50"
							>
								<div class="flex items-center space-x-2 text-gray-500 dark:text-gray-400">
									<div class="h-3 w-3 animate-bounce rounded-full bg-blue-500"></div>
									<div
										class="h-3 w-3 animate-bounce rounded-full bg-blue-500"
										style="animation-delay: 0.2s"
									></div>
									<div
										class="h-3 w-3 animate-bounce rounded-full bg-blue-500"
										style="animation-delay: 0.4s"
									></div>
								</div>
							</div>
						</div>
					</div>
				{/if}
			</div>
		{/if}
	</main>

	<!-- Input area -->
	<footer
		class="fixed right-0 bottom-0 left-0 bg-gradient-to-t from-white via-white/90 to-transparent px-4 pt-12 pb-6 dark:from-gray-900 dark:via-gray-900/90"
	>
		<div class="mx-auto max-w-3xl">
			<div
				class="flex overflow-hidden rounded-xl border border-gray-300 bg-white shadow-lg dark:border-gray-600 dark:bg-gray-800"
			>
				<textarea
					bind:value={inputText}
					onkeydown={handleKeydown}
					placeholder="Message your LLM..."
					class="max-h-40 min-h-[60px] flex-1 resize-none border-0 bg-transparent px-4 py-3 text-gray-900 placeholder-gray-500 focus:ring-0 dark:text-white dark:placeholder-gray-400"
					disabled={loading}
				/>
				<button
					onclick={sendMessage}
					class="m-2 flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-r from-blue-600 to-indigo-700 px-4 py-2 text-white disabled:opacity-50"
					disabled={!inputText.trim() || loading}
				>
					{#if loading}
						<i class="fa-solid fa-ellipsis animate-pulse"></i>
					{:else}
						<i class="fa-solid fa-paper-plane"></i>
					{/if}
				</button>
			</div>
		</div>
	</footer>
</div>

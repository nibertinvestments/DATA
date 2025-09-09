# Advanced Elixir Concurrent Programming Examples
# This module demonstrates intermediate to advanced Elixir concepts including:
# - Actor model with GenServer and OTP
# - Supervision trees and fault tolerance
# - Distributed computing patterns
# - Concurrent data processing
# - Phoenix web applications
# - Pattern matching and functional programming

defmodule AdvancedElixir do
  @moduledoc """
  Comprehensive examples of advanced Elixir programming patterns
  focusing on concurrency, OTP, and distributed systems.
  """

  # Advanced Data Structures and Pattern Matching
  # =============================================

  @doc """
  Binary tree implementation with pattern matching
  """
  defmodule BinaryTree do
    defstruct value: nil, left: nil, right: nil

    @type t :: %__MODULE__{
      value: any(),
      left: t() | nil,
      right: t() | nil
    }

    @spec insert(t() | nil, any()) :: t()
    def insert(nil, value), do: %__MODULE__{value: value}
    
    def insert(%__MODULE__{value: root_value, left: left, right: right}, value) do
      cond do
        value <= root_value ->
          %__MODULE__{value: root_value, left: insert(left, value), right: right}
        value > root_value ->
          %__MODULE__{value: root_value, left: left, right: insert(right, value)}
      end
    end

    @spec search(t() | nil, any()) :: boolean()
    def search(nil, _value), do: false
    
    def search(%__MODULE__{value: value}, value), do: true
    
    def search(%__MODULE__{value: root_value, left: left, right: right}, value) do
      cond do
        value < root_value -> search(left, value)
        value > root_value -> search(right, value)
      end
    end

    @spec inorder(t() | nil) :: [any()]
    def inorder(nil), do: []
    
    def inorder(%__MODULE__{value: value, left: left, right: right}) do
      inorder(left) ++ [value] ++ inorder(right)
    end

    @spec depth(t() | nil) :: non_neg_integer()
    def depth(nil), do: 0
    
    def depth(%__MODULE__{left: left, right: right}) do
      1 + max(depth(left), depth(right))
    end
  end

  # GenServer Implementation for Stateful Processes
  # ===============================================

  @doc """
  Bank account GenServer with thread-safe operations
  """
  defmodule BankAccount do
    use GenServer

    # Client API
    def start_link(initial_balance) when is_number(initial_balance) do
      GenServer.start_link(__MODULE__, initial_balance)
    end

    def balance(pid) do
      GenServer.call(pid, :balance)
    end

    def deposit(pid, amount) when is_number(amount) and amount > 0 do
      GenServer.call(pid, {:deposit, amount})
    end

    def withdraw(pid, amount) when is_number(amount) and amount > 0 do
      GenServer.call(pid, {:withdraw, amount})
    end

    def transfer(from_pid, to_pid, amount) when is_number(amount) and amount > 0 do
      GenServer.call(from_pid, {:transfer, to_pid, amount})
    end

    def transaction_history(pid) do
      GenServer.call(pid, :history)
    end

    # Server Callbacks
    @impl true
    def init(initial_balance) do
      state = %{
        balance: initial_balance,
        history: [{:initial_deposit, initial_balance, DateTime.utc_now()}]
      }
      {:ok, state}
    end

    @impl true
    def handle_call(:balance, _from, %{balance: balance} = state) do
      {:reply, balance, state}
    end

    @impl true
    def handle_call({:deposit, amount}, _from, %{balance: balance, history: history} = state) do
      new_balance = balance + amount
      new_history = [{:deposit, amount, DateTime.utc_now()} | history]
      new_state = %{state | balance: new_balance, history: new_history}
      {:reply, {:ok, new_balance}, new_state}
    end

    @impl true
    def handle_call({:withdraw, amount}, _from, %{balance: balance, history: history} = state) do
      if balance >= amount do
        new_balance = balance - amount
        new_history = [{:withdraw, amount, DateTime.utc_now()} | history]
        new_state = %{state | balance: new_balance, history: new_history}
        {:reply, {:ok, new_balance}, new_state}
      else
        {:reply, {:error, :insufficient_funds}, state}
      end
    end

    @impl true
    def handle_call({:transfer, to_pid, amount}, _from, %{balance: balance} = state) do
      if balance >= amount do
        case withdraw(self(), amount) do
          {:ok, _} ->
            case deposit(to_pid, amount) do
              {:ok, _} ->
                {:reply, {:ok, :transfer_completed}, state}
              {:error, reason} ->
                # Rollback the withdrawal
                deposit(self(), amount)
                {:reply, {:error, reason}, state}
            end
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
      else
        {:reply, {:error, :insufficient_funds}, state}
      end
    end

    @impl true
    def handle_call(:history, _from, %{history: history} = state) do
      {:reply, Enum.reverse(history), state}
    end
  end

  # Supervision Tree Implementation
  # ==============================

  @doc """
  Bank supervisor for fault tolerance
  """
  defmodule BankSupervisor do
    use Supervisor

    def start_link(opts) do
      Supervisor.start_link(__MODULE__, :ok, opts)
    end

    @impl true
    def init(:ok) do
      children = [
        {BankAccountRegistry, name: BankAccountRegistry},
        {Task.Supervisor, name: BankTaskSupervisor}
      ]

      Supervisor.init(children, strategy: :one_for_one)
    end
  end

  @doc """
  Registry for managing bank accounts
  """
  defmodule BankAccountRegistry do
    use GenServer

    # Client API
    def start_link(opts) do
      GenServer.start_link(__MODULE__, :ok, opts)
    end

    def create_account(registry_pid, account_id, initial_balance) do
      GenServer.call(registry_pid, {:create_account, account_id, initial_balance})
    end

    def get_account(registry_pid, account_id) do
      GenServer.call(registry_pid, {:get_account, account_id})
    end

    def list_accounts(registry_pid) do
      GenServer.call(registry_pid, :list_accounts)
    end

    # Server Callbacks
    @impl true
    def init(:ok) do
      {:ok, %{}}
    end

    @impl true
    def handle_call({:create_account, account_id, initial_balance}, _from, accounts) do
      if Map.has_key?(accounts, account_id) do
        {:reply, {:error, :account_exists}, accounts}
      else
        case BankAccount.start_link(initial_balance) do
          {:ok, pid} ->
            Process.monitor(pid)
            new_accounts = Map.put(accounts, account_id, pid)
            {:reply, {:ok, pid}, new_accounts}
          {:error, reason} ->
            {:reply, {:error, reason}, accounts}
        end
      end
    end

    @impl true
    def handle_call({:get_account, account_id}, _from, accounts) do
      case Map.get(accounts, account_id) do
        nil -> {:reply, {:error, :account_not_found}, accounts}
        pid -> {:reply, {:ok, pid}, accounts}
      end
    end

    @impl true
    def handle_call(:list_accounts, _from, accounts) do
      account_list = Enum.map(accounts, fn {id, pid} -> 
        {id, BankAccount.balance(pid)}
      end)
      {:reply, account_list, accounts}
    end

    @impl true
    def handle_info({:DOWN, _ref, :process, pid, _reason}, accounts) do
      # Remove the crashed account from the registry
      new_accounts = 
        accounts
        |> Enum.reject(fn {_id, account_pid} -> account_pid == pid end)
        |> Map.new()
      
      {:noreply, new_accounts}
    end
  end

  # Concurrent Data Processing Pipeline
  # ==================================

  @doc """
  Concurrent data processing using Task and Flow
  """
  defmodule DataProcessor do
    
    @spec process_data([any()], keyword()) :: [any()]
    def process_data(data, opts \\ []) do
      concurrency = Keyword.get(opts, :concurrency, System.schedulers_online())
      timeout = Keyword.get(opts, :timeout, 5000)

      data
      |> Task.async_stream(&expensive_computation/1, 
           max_concurrency: concurrency, 
           timeout: timeout,
           on_timeout: :kill_task)
      |> Enum.map(fn
        {:ok, result} -> result
        {:exit, :timeout} -> {:error, :timeout}
      end)
    end

    @spec parallel_map([any()], fun(), keyword()) :: [any()]
    def parallel_map(collection, func, opts \\ []) do
      chunk_size = Keyword.get(opts, :chunk_size, 100)
      
      collection
      |> Enum.chunk_every(chunk_size)
      |> Task.async_stream(fn chunk ->
        Enum.map(chunk, func)
      end, ordered: false)
      |> Enum.reduce([], fn {:ok, results}, acc ->
        acc ++ results
      end)
    end

    @spec pipeline_process([any()]) :: [any()]
    def pipeline_process(data) do
      data
      |> Stream.map(&normalize_data/1)
      |> Stream.filter(&valid_data?/1)
      |> Stream.map(&transform_data/1)
      |> Stream.chunk_every(100)
      |> Task.async_stream(&batch_process/1, max_concurrency: 4)
      |> Stream.map(fn {:ok, result} -> result end)
      |> Enum.to_list()
      |> List.flatten()
    end

    # Private helper functions
    defp expensive_computation(item) do
      # Simulate expensive computation
      :timer.sleep(Enum.random(10..100))
      item * 2
    end

    defp normalize_data(data) when is_binary(data), do: String.trim(data)
    defp normalize_data(data), do: data

    defp valid_data?(data) when is_binary(data), do: String.length(data) > 0
    defp valid_data?(_), do: true

    defp transform_data(data) when is_binary(data), do: String.upcase(data)
    defp transform_data(data), do: data

    defp batch_process(batch) do
      # Simulate batch processing
      :timer.sleep(50)
      Enum.map(batch, fn item -> "Processed: #{item}" end)
    end
  end

  # Distributed Computing with Nodes
  # ================================

  @doc """
  Distributed computing coordinator
  """
  defmodule DistributedCoordinator do
    use GenServer

    defstruct [:nodes, :tasks, :results]

    # Client API
    def start_link(opts \\ []) do
      GenServer.start_link(__MODULE__, opts, name: __MODULE__)
    end

    def add_node(node) do
      GenServer.call(__MODULE__, {:add_node, node})
    end

    def distribute_work(work_items) do
      GenServer.call(__MODULE__, {:distribute_work, work_items}, 30_000)
    end

    def get_cluster_status do
      GenServer.call(__MODULE__, :cluster_status)
    end

    # Server Callbacks
    @impl true
    def init(_opts) do
      # Monitor node connections
      :net_kernel.monitor_nodes(true)
      
      state = %__MODULE__{
        nodes: [Node.self()],
        tasks: %{},
        results: []
      }
      
      {:ok, state}
    end

    @impl true
    def handle_call({:add_node, node}, _from, %{nodes: nodes} = state) do
      case Node.connect(node) do
        true ->
          new_nodes = [node | nodes] |> Enum.uniq()
          {:reply, :ok, %{state | nodes: new_nodes}}
        false ->
          {:reply, {:error, :connection_failed}, state}
      end
    end

    @impl true
    def handle_call({:distribute_work, work_items}, _from, %{nodes: nodes} = state) do
      # Distribute work across available nodes
      work_chunks = Enum.chunk_every(work_items, div(length(work_items), length(nodes)) + 1)
      
      tasks = 
        nodes
        |> Enum.zip(work_chunks)
        |> Enum.map(fn {node, chunk} ->
          Task.Supervisor.async({TaskSupervisor, node}, fn ->
            process_work_chunk(chunk)
          end)
        end)

      # Wait for all tasks to complete
      results = 
        tasks
        |> Task.yield_many(30_000)
        |> Enum.map(fn
          {task, {:ok, result}} -> result
          {task, nil} -> 
            Task.shutdown(task, :brutal_kill)
            {:error, :timeout}
        end)

      {:reply, {:ok, List.flatten(results)}, state}
    end

    @impl true
    def handle_call(:cluster_status, _from, %{nodes: nodes} = state) do
      status = %{
        connected_nodes: nodes,
        active_nodes: Enum.filter(nodes, &Node.ping(&1) == :pong),
        total_processes: :erlang.system_info(:process_count)
      }
      
      {:reply, status, state}
    end

    @impl true
    def handle_info({:nodeup, node}, %{nodes: nodes} = state) do
      new_nodes = [node | nodes] |> Enum.uniq()
      {:noreply, %{state | nodes: new_nodes}}
    end

    @impl true
    def handle_info({:nodedown, node}, %{nodes: nodes} = state) do
      new_nodes = List.delete(nodes, node)
      {:noreply, %{state | nodes: new_nodes}}
    end

    # Private functions
    defp process_work_chunk(chunk) do
      Enum.map(chunk, fn item ->
        # Simulate work processing
        :timer.sleep(Enum.random(10..100))
        "Node #{Node.self()}: Processed #{item}"
      end)
    end
  end

  # Real-time Communication with Phoenix Channels
  # =============================================

  @doc """
  Real-time chat system using Phoenix Channels
  """
  defmodule ChatSystem do
    
    # Message structure
    defmodule Message do
      defstruct [:id, :user, :content, :timestamp, :room]
      
      @type t :: %__MODULE__{
        id: String.t(),
        user: String.t(),
        content: String.t(),
        timestamp: DateTime.t(),
        room: String.t()
      }
    end

    # Chat room GenServer
    defmodule ChatRoom do
      use GenServer

      defstruct [:name, :messages, :users, :created_at]

      # Client API
      def start_link(room_name) do
        GenServer.start_link(__MODULE__, room_name, name: via_tuple(room_name))
      end

      def join_room(room_name, user) do
        GenServer.call(via_tuple(room_name), {:join, user})
      end

      def leave_room(room_name, user) do
        GenServer.call(via_tuple(room_name), {:leave, user})
      end

      def send_message(room_name, user, content) do
        GenServer.call(via_tuple(room_name), {:message, user, content})
      end

      def get_messages(room_name, limit \\ 50) do
        GenServer.call(via_tuple(room_name), {:get_messages, limit})
      end

      def get_users(room_name) do
        GenServer.call(via_tuple(room_name), :get_users)
      end

      # Server Callbacks
      @impl true
      def init(room_name) do
        state = %__MODULE__{
          name: room_name,
          messages: [],
          users: MapSet.new(),
          created_at: DateTime.utc_now()
        }
        
        {:ok, state}
      end

      @impl true
      def handle_call({:join, user}, _from, %{users: users} = state) do
        if MapSet.member?(users, user) do
          {:reply, {:error, :already_joined}, state}
        else
          new_users = MapSet.put(users, user)
          broadcast_user_joined(state.name, user)
          {:reply, :ok, %{state | users: new_users}}
        end
      end

      @impl true
      def handle_call({:leave, user}, _from, %{users: users} = state) do
        new_users = MapSet.delete(users, user)
        broadcast_user_left(state.name, user)
        {:reply, :ok, %{state | users: new_users}}
      end

      @impl true
      def handle_call({:message, user, content}, _from, %{messages: messages, users: users} = state) do
        if MapSet.member?(users, user) do
          message = %Message{
            id: generate_id(),
            user: user,
            content: content,
            timestamp: DateTime.utc_now(),
            room: state.name
          }
          
          new_messages = [message | messages] |> Enum.take(1000)  # Keep last 1000 messages
          broadcast_message(state.name, message)
          
          {:reply, {:ok, message}, %{state | messages: new_messages}}
        else
          {:reply, {:error, :not_in_room}, state}
        end
      end

      @impl true
      def handle_call({:get_messages, limit}, _from, %{messages: messages} = state) do
        recent_messages = 
          messages
          |> Enum.take(limit)
          |> Enum.reverse()
        
        {:reply, recent_messages, state}
      end

      @impl true
      def handle_call(:get_users, _from, %{users: users} = state) do
        {:reply, MapSet.to_list(users), state}
      end

      # Private functions
      defp via_tuple(room_name) do
        {:via, Registry, {ChatRoomRegistry, room_name}}
      end

      defp generate_id do
        :crypto.strong_rand_bytes(16) |> Base.encode16()
      end

      defp broadcast_message(room, message) do
        # In a real Phoenix app, this would broadcast to channels
        IO.puts("Broadcasting message in #{room}: #{message.user}: #{message.content}")
      end

      defp broadcast_user_joined(room, user) do
        IO.puts("User #{user} joined room #{room}")
      end

      defp broadcast_user_left(room, user) do
        IO.puts("User #{user} left room #{room}")
      end
    end
  end

  # Functional Programming Patterns
  # ===============================

  @doc """
  Advanced functional programming utilities
  """
  defmodule FunctionalUtils do
    
    # Monadic operations
    def bind(value, func) when is_function(func, 1) do
      case value do
        {:ok, val} -> func.(val)
        {:error, _} = error -> error
        nil -> nil
        val -> func.(val)
      end
    end

    def map_result(value, func) when is_function(func, 1) do
      case value do
        {:ok, val} -> {:ok, func.(val)}
        {:error, _} = error -> error
      end
    end

    # Lens-like operations for nested data
    def get_in_safe(data, path) do
      try do
        {:ok, get_in(data, path)}
      rescue
        _ -> {:error, :invalid_path}
      end
    end

    def put_in_safe(data, path, value) do
      try do
        {:ok, put_in(data, path, value)}
      rescue
        _ -> {:error, :invalid_path}
      end
    end

    # Currying support
    def curry(func, arity) do
      curry(func, arity, [])
    end

    defp curry(func, 0, args) do
      apply(func, Enum.reverse(args))
    end

    defp curry(func, arity, args) when arity > 0 do
      fn arg ->
        curry(func, arity - 1, [arg | args])
      end
    end

    # Composition
    def compose(f, g) do
      fn x -> f.(g.(x)) end
    end

    def pipe_through(value, functions) do
      Enum.reduce(functions, value, fn func, acc -> func.(acc) end)
    end

    # Memoization
    def memoize(func) when is_function(func, 1) do
      agent = Agent.start_link(fn -> %{} end)
      
      fn arg ->
        case Agent.get(agent, &Map.get(&1, arg)) do
          nil ->
            result = func.(arg)
            Agent.update(agent, &Map.put(&1, arg, result))
            result
          cached_result ->
            cached_result
        end
      end
    end
  end

  # Error Handling and Resilience Patterns
  # ======================================

  @doc """
  Circuit breaker pattern implementation
  """
  defmodule CircuitBreaker do
    use GenServer

    defstruct [
      :name,
      :failure_threshold,
      :timeout,
      :state,
      :failure_count,
      :last_failure_time
    ]

    @type state :: :closed | :open | :half_open
    @type t :: %__MODULE__{
      name: String.t(),
      failure_threshold: pos_integer(),
      timeout: pos_integer(),
      state: state(),
      failure_count: non_neg_integer(),
      last_failure_time: DateTime.t() | nil
    }

    # Client API
    def start_link(opts) do
      name = Keyword.fetch!(opts, :name)
      GenServer.start_link(__MODULE__, opts, name: via_tuple(name))
    end

    def call(circuit_name, func) when is_function(func, 0) do
      GenServer.call(via_tuple(circuit_name), {:call, func})
    end

    def get_state(circuit_name) do
      GenServer.call(via_tuple(circuit_name), :get_state)
    end

    def reset(circuit_name) do
      GenServer.call(via_tuple(circuit_name), :reset)
    end

    # Server Callbacks
    @impl true
    def init(opts) do
      state = %__MODULE__{
        name: Keyword.fetch!(opts, :name),
        failure_threshold: Keyword.get(opts, :failure_threshold, 5),
        timeout: Keyword.get(opts, :timeout, 60_000),
        state: :closed,
        failure_count: 0,
        last_failure_time: nil
      }
      
      {:ok, state}
    end

    @impl true
    def handle_call({:call, func}, _from, state) do
      case state.state do
        :closed ->
          execute_call(func, state)
        
        :open ->
          if should_attempt_reset?(state) do
            new_state = %{state | state: :half_open}
            execute_call(func, new_state)
          else
            {:reply, {:error, :circuit_open}, state}
          end
        
        :half_open ->
          execute_call(func, state)
      end
    end

    @impl true
    def handle_call(:get_state, _from, state) do
      {:reply, state, state}
    end

    @impl true
    def handle_call(:reset, _from, state) do
      new_state = %{state | state: :closed, failure_count: 0, last_failure_time: nil}
      {:reply, :ok, new_state}
    end

    # Private functions
    defp via_tuple(name) do
      {:via, Registry, {CircuitBreakerRegistry, name}}
    end

    defp execute_call(func, state) do
      try do
        result = func.()
        success_state = handle_success(state)
        {:reply, {:ok, result}, success_state}
      rescue
        error ->
          failure_state = handle_failure(state, error)
          {:reply, {:error, error}, failure_state}
      end
    end

    defp handle_success(%{state: :half_open} = state) do
      %{state | state: :closed, failure_count: 0, last_failure_time: nil}
    end

    defp handle_success(state), do: state

    defp handle_failure(state, _error) do
      new_failure_count = state.failure_count + 1
      new_last_failure_time = DateTime.utc_now()
      
      new_state = 
        if new_failure_count >= state.failure_threshold do
          :open
        else
          state.state
        end

      %{state | 
        failure_count: new_failure_count,
        last_failure_time: new_last_failure_time,
        state: new_state
      }
    end

    defp should_attempt_reset?(%{last_failure_time: nil}), do: true
    
    defp should_attempt_reset?(%{last_failure_time: last_failure, timeout: timeout}) do
      DateTime.diff(DateTime.utc_now(), last_failure, :millisecond) >= timeout
    end
  end

  # Performance Monitoring and Metrics
  # ==================================

  @doc """
  Performance monitoring utilities
  """
  defmodule PerformanceMonitor do
    
    def measure(name, func) when is_function(func, 0) do
      start_time = System.monotonic_time(:microsecond)
      
      try do
        result = func.()
        end_time = System.monotonic_time(:microsecond)
        duration = end_time - start_time
        
        log_metric(name, duration, :success)
        {:ok, result, duration}
      rescue
        error ->
          end_time = System.monotonic_time(:microsecond)
          duration = end_time - start_time
          
          log_metric(name, duration, :error)
          {:error, error, duration}
      end
    end

    def benchmark(name, func, iterations \\ 1000) do
      measurements = 
        1..iterations
        |> Enum.map(fn _ ->
          {_, _, duration} = measure(name, func)
          duration
        end)

      %{
        name: name,
        iterations: iterations,
        min: Enum.min(measurements),
        max: Enum.max(measurements),
        avg: div(Enum.sum(measurements), length(measurements)),
        median: median(measurements)
      }
    end

    defp log_metric(name, duration, status) do
      IO.puts("METRIC: #{name} - Duration: #{duration}μs - Status: #{status}")
    end

    defp median(list) do
      sorted = Enum.sort(list)
      length = length(sorted)
      
      if rem(length, 2) == 0 do
        (Enum.at(sorted, div(length, 2) - 1) + Enum.at(sorted, div(length, 2))) / 2
      else
        Enum.at(sorted, div(length, 2))
      end
    end
  end

  # Example Usage and Testing
  # ========================

  @doc """
  Comprehensive demonstration of all features
  """
  def demo do
    IO.puts("=== Advanced Elixir Programming Examples ===\n")

    # Binary Tree Demo
    IO.puts("1. Binary Search Tree:")
    tree = 
      [5, 3, 8, 1, 4, 7, 9]
      |> Enum.reduce(nil, &BinaryTree.insert(&2, &1))
    
    IO.puts("   Tree inorder: #{inspect(BinaryTree.inorder(tree))}")
    IO.puts("   Contains 4: #{BinaryTree.search(tree, 4)}")
    IO.puts("   Tree depth: #{BinaryTree.depth(tree)}")

    # Concurrent Processing Demo
    IO.puts("\n2. Concurrent Data Processing:")
    data = 1..100 |> Enum.to_list()
    
    {result, duration} = PerformanceMonitor.measure("parallel_processing", fn ->
      DataProcessor.process_data(data, concurrency: 4)
    end)
    
    case result do
      {:ok, processed_data} ->
        IO.puts("   Processed #{length(processed_data)} items in #{duration}μs")
      {:error, reason} ->
        IO.puts("   Processing failed: #{inspect(reason)}")
    end

    # GenServer Demo
    IO.puts("\n3. Bank Account GenServer:")
    {:ok, account1} = BankAccount.start_link(1000)
    {:ok, account2} = BankAccount.start_link(500)
    
    IO.puts("   Account 1 balance: #{BankAccount.balance(account1)}")
    IO.puts("   Account 2 balance: #{BankAccount.balance(account2)}")
    
    BankAccount.transfer(account1, account2, 200)
    
    IO.puts("   After transfer:")
    IO.puts("   Account 1 balance: #{BankAccount.balance(account1)}")
    IO.puts("   Account 2 balance: #{BankAccount.balance(account2)}")

    # Circuit Breaker Demo
    IO.puts("\n4. Circuit Breaker Pattern:")
    
    # This would need proper setup in a real application
    IO.puts("   Circuit breaker protects against cascading failures")
    IO.puts("   States: closed (normal), open (failing), half-open (testing)")

    # Functional Programming Demo
    IO.puts("\n5. Functional Programming Utilities:")
    
    add_ten = fn x -> x + 10 end
    multiply_by_two = fn x -> x * 2 end
    composed_func = FunctionalUtils.compose(multiply_by_two, add_ten)
    
    result = composed_func.(5)
    IO.puts("   Composed function (5 + 10) * 2 = #{result}")
    
    pipeline_result = FunctionalUtils.pipe_through(5, [add_ten, multiply_by_two])
    IO.puts("   Pipeline result: #{pipeline_result}")

    IO.puts("\n=== Elixir Demo Complete ===")
  end
end

# Main execution
if :erlang.system_info(:schedulers_online) > 0 do
  AdvancedElixir.demo()
end
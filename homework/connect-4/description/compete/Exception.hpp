#ifndef _EXCEPTIONS_HPP_

#define _EXCEPTIONS_HPP_

#include <signal.h>

#include <exception>
#include <string>

namespace Exception
{

class BaseException : public std::exception
{
public:
    virtual const char *what() const noexcept
    {
        return "base_exception";
    }
};

class FloatingPointException : public BaseException
{
public:
    virtual const char *what() const noexcept
    {
        return "floating_point_exception";
    }
};

class SegmentFaultException : public BaseException
{
public:
    virtual const char *what() const noexcept
    {
        return "segment_fault_exception";
    }
};

class BusErrorException : public BaseException
{
public:
    virtual const char *what() const noexcept
    {
        return "bus_error_exception";
    }
};

class DoubleFreeException : public BaseException
{
public:
    virtual const char *what() const noexcept
    {
        return "double_free_exception";
    }
};

class __signal_init
{
    static void UnblockSignal(int sig)
    {
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, sig);
        sigprocmask(SIG_UNBLOCK, &set, nullptr);
    }
    static void SignalExceptionHandler(int sig)
    {
        UnblockSignal(sig);

        switch (sig)
        {
        case SIGFPE:
            throw FloatingPointException();
        case SIGSEGV:
            throw SegmentFaultException();
        case SIGBUS:
            throw BusErrorException();
        case SIGABRT:
            throw DoubleFreeException();
        }
    }

    static int __compitable;

    static int _()
    {
        signal(SIGFPE, SignalExceptionHandler);
        signal(SIGSEGV, SignalExceptionHandler);
        signal(SIGBUS, SignalExceptionHandler);
        signal(SIGABRT, SignalExceptionHandler);

        signal(SIGPIPE, SIG_IGN);

        return 0;
    }
};

int __signal_init::__compitable = __signal_init::_();

class Error
{
    std::string info;

public:
    Error(const char *info)
        : info(info)
    {
    }

    Error(std::string info)
        : info(info)
    {
    }

    Error(int err)
    {
        if (err)
            info = std::to_string(err);
    }

    virtual const char *what()
    {
        return info.c_str();
    }

    operator bool()
    {
        return info.size();
    }

    operator std::string()
    {
        return info;
    }
};
} // namespace Exception

#endif

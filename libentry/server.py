#!/usr/bin/env python3

import os
from socketserver import ThreadingMixIn
from threading import Thread
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

__all__ = [
    'XMLRPCServerMixIn'
]


class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


class MultRequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', '/RPC3')


class XMLRPCServerMixIn:

    def serve_forever(self, addr, daemon=False):
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
            if rank > 0:
                return

        with ThreadXMLRPCServer(addr, requestHandler=MultRequestHandler, allow_none=True) as server:
            server.register_introspection_functions()
            server.register_multicall_functions()
            server.register_instance(self)
            if daemon:
                if hasattr(self, 'serve_thread'):
                    raise RuntimeError('Server is already running.')
                serve_thread = Thread(target=server.serve_forever, daemon=daemon)
                serve_thread.start()
                setattr(self, 'serve_thread', serve_thread)
            else:
                server.serve_forever()
